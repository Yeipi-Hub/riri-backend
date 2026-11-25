"""RIRI API: registro bidireccional y consulta de posibles coincidencias.

Dependencias: pip install fastapi uvicorn python-multipart boto3 firebase-admin pillow
Variables de entorno: GOOGLE_APPLICATION_CREDENTIALS, AWS_REGION (opcional),
AWS_REKOGNITION_COLLECTION (opcional), RIRI_API_KEY.
AWS: usar IAM role o AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY en entorno.

IMPORTANTE: prototipo para hackathon. La clave de API compartida no sustituye
una autenticación individual y autorización robusta en producción.
"""
import base64
import io
import os
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status

import boto3
import firebase_admin
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import transactional
from PIL import Image, ImageOps, UnidentifiedImageError

REGION_NAME = os.getenv("AWS_REGION", "us-east-1")
COLLECTION_ID = os.getenv("AWS_REKOGNITION_COLLECTION", "riri-personas-desaparecidas")
API_KEY = os.getenv("RIRI_API_KEY")
MATCH_THRESHOLD = 80.0
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_STORED_IMAGE_BYTES = 600_000  # margen bajo el límite de 1 MiB por documento Firestore
MAX_AVISTAMIENTOS_COMPARE = 300  # límite de seguridad para demo; paginar para producción

if not firebase_admin._apps:
    service_account = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if service_account:
        firebase_admin.initialize_app(credentials.Certificate(service_account))
    else:
        firebase_admin.initialize_app()  # Application Default Credentials

db = firestore.client()
rekognition = boto3.client("rekognition", region_name=REGION_NAME)
app = FastAPI(title="RIRI Biometric API", version="2.0")


def require_admin_key(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Configurar RIRI_API_KEY en el servidor")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Credencial de acceso inválida")


def imagen_jpeg(original: bytes) -> bytes:
    if len(original) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Imagen demasiado grande (máximo 8 MB)")
    try:
        with Image.open(io.BytesIO(original)) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            # Reducir para almacenar en Firestore sin exceder tamaño del documento.
            for max_side, quality in [(960, 80), (720, 72), (540, 65), (420, 55)]:
                img = im.copy()
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                img.save(output, format="JPEG", quality=quality, optimize=True)
                result = output.getvalue()
                if len(result) <= MAX_STORED_IMAGE_BYTES:
                    return result
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Archivo de imagen inválido")
    raise HTTPException(status_code=413, detail="La imagen no se pudo comprimir suficientemente")


async def leer_imagen(file: UploadFile) -> bytes:
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/heic", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Formato de archivo no permitido")
    original = await file.read(MAX_UPLOAD_BYTES + 1)
    if not original:
        raise HTTPException(status_code=400, detail="Imagen vacía")
    return imagen_jpeg(original)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def ahora():
    return datetime.now(timezone.utc).isoformat()


@transactional
def siguiente_folio(transaction, counter_ref):
    snap = counter_ref.get(transaction=transaction)
    current = (snap.to_dict() or {}).get("ultimo_numero", 0) if snap.exists else 0
    numero = int(current) + 1
    transaction.set(counter_ref, {"ultimo_numero": numero})
    return f"RIRI-{numero:03d}"


def crear_folio():
    ref = db.collection("configuracion").document("contador_folios")
    return siguiente_folio(db.transaction(), ref)


def coincidencia_guardada(folio: str, avistamiento_id: str, similitud: float, origen: str):
    """Una relación por pareja; la comparación se puede repetir sin duplicar registros."""
    id_coincidencia = f"{folio}__{avistamiento_id}"
    ref = db.collection("coincidencias").document(id_coincidencia)
    existing = ref.get()
    ref.set({
        "folio": folio,
        "avistamiento_id": avistamiento_id,
        "porcentaje_similitud": round(float(similitud), 2),
        "similarity": round(float(similitud), 2),
        "origen": origen,
        "estado": "pendiente_verificacion",
        "actualizado_en": ahora(),
        "creado_en": (existing.to_dict() or {}).get("creado_en", ahora()) if existing.exists else ahora(),
    }, merge=True)
    return id_coincidencia


def datos_resultado(doc):
    """Une ambas colecciones para que Flutter reciba fotos y metadatos."""
    match = doc.to_dict() or {}
    desaparecido = db.collection("desaparecidos").document(match["folio"]).get()
    avistamiento = db.collection("avistamientos").document(match["avistamiento_id"]).get()
    d = desaparecido.to_dict() if desaparecido.exists else {}
    a = avistamiento.to_dict() if avistamiento.exists else {}
    return {
        "id": doc.id,
        "folio": match["folio"],
        "nombre": d.get("nombre"),
        "edad": d.get("edad"),
        "imagen_desaparecido_base64": d.get("imagen_base64"),
        "avistamiento_id": match["avistamiento_id"],
        "imagen_avistamiento_base64": a.get("imagen_base64"),
        "porcentaje_similitud": match.get("porcentaje_similitud", match.get("similarity")),
        "estado": match.get("estado", "pendiente_verificacion"),
        "creado_en": match.get("creado_en"),
    }


def buscar_desaparecidos(image_bytes: bytes, avistamiento_id: str):
    try:
        response = rekognition.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={"Bytes": image_bytes},
            MaxFaces=10,
            FaceMatchThreshold=MATCH_THRESHOLD,
        )
    except rekognition.exceptions.InvalidParameterException:
        raise HTTPException(status_code=422, detail="No se detectó un rostro válido en la fotografía")
    matches = []
    for match in response.get("FaceMatches", []):
        folio = match["Face"].get("ExternalImageId")
        similarity = float(match["Similarity"])
        if not folio or not db.collection("desaparecidos").document(folio).get().exists:
            continue
        id_match = coincidencia_guardada(folio, avistamiento_id, similarity, "nuevo_avistamiento")
        matches.append(datos_resultado(db.collection("coincidencias").document(id_match).get()))
    return matches


def comparar_avistamientos_anteriores(folio: str, image_bytes: bytes):
    """CompareFaces compara la foto nueva contra fotos de avistamientos previos.
    Para bases grandes, mover avistamientos a un índice de búsqueda escalable.
    """
    matches = []
    docs = db.collection("avistamientos").limit(MAX_AVISTAMIENTOS_COMPARE + 1).stream()
    avistamientos = list(docs)
    if len(avistamientos) > MAX_AVISTAMIENTOS_COMPARE:
        raise HTTPException(
            status_code=409,
            detail="Hay más avistamientos que el límite de la demo; se requiere búsqueda paginada/índice escalable",
        )
    for doc in avistamientos:
        data = doc.to_dict() or {}
        encoded = data.get("imagen_base64")
        if not encoded:
            continue
        try:
            result = rekognition.compare_faces(
                SourceImage={"Bytes": image_bytes},
                TargetImage={"Bytes": base64.b64decode(encoded)},
                SimilarityThreshold=MATCH_THRESHOLD,
            )
            for match in result.get("FaceMatches", []):
                similarity = float(match["Similarity"])
                id_match = coincidencia_guardada(folio, doc.id, similarity, "nuevo_desaparecido")
                matches.append(datos_resultado(db.collection("coincidencias").document(id_match).get()))
                break
        except rekognition.exceptions.InvalidParameterException:
            continue  # avistamiento anterior sin rostro detectable
    return sorted(matches, key=lambda x: x["porcentaje_similitud"], reverse=True)


@app.get("/")
def inicio():
    return {"status": "RIRI API activa", "version": "2.0"}


@app.post("/api/desaparecidos/registrar", dependencies=[Depends(require_admin_key)])
async def registrar_desaparecido(nombre: str = Form(...), edad: int = Form(...), file: UploadFile = File(...)):
    if not nombre.strip() or not 0 <= edad <= 120:
        raise HTTPException(status_code=422, detail="Nombre o edad inválidos")
    image_bytes = await leer_imagen(file)
    folio = crear_folio()
    try:
        response = rekognition.index_faces(
            CollectionId=COLLECTION_ID,
            Image={"Bytes": image_bytes},
            ExternalImageId=folio,
            MaxFaces=1,
            DetectionAttributes=["DEFAULT"],
        )
    except rekognition.exceptions.InvalidParameterException:
        raise HTTPException(status_code=422, detail="No se pudo detectar un rostro")
    records = response.get("FaceRecords", [])
    if not records:
        raise HTTPException(status_code=422, detail="No se detectó un rostro válido")
    face_id = records[0]["Face"]["FaceId"]
    db.collection("desaparecidos").document(folio).set({
        "folio": folio,
        "nombre": nombre.strip(),
        "edad": edad,
        "face_id": face_id,
        "imagen_base64": b64(image_bytes),
        "status": "activo",
        "timestamp": firestore.SERVER_TIMESTAMP,
    })
    # Comparación inversa: la nueva persona desaparecida contra avistamientos ya guardados.
    # Si falla la comparación, el registro sigue existiendo y puede reintentarse.
    try:
        matches = comparar_avistamientos_anteriores(folio, image_bytes)
        warning = None
    except Exception as exc:
        matches = []
        warning = f"Registro guardado; comparación con avistamientos pendiente: {type(exc).__name__}"
    return {
        "status": "success", "folio": folio, "face_id": face_id,
        "coincidencia": bool(matches), "coincidencias": matches,
        "total_coincidencias": len(matches), "advertencia": warning,
    }


@app.post("/api/avistamientos/buscar", dependencies=[Depends(require_admin_key)])
async def registrar_avistamiento(file: UploadFile = File(...)):
    image_bytes = await leer_imagen(file)
    avistamiento_ref = db.collection("avistamientos").document()
    avistamiento_ref.set({
        "avistamiento_id": avistamiento_ref.id,
        "imagen_base64": b64(image_bytes),
        "timestamp": firestore.SERVER_TIMESTAMP,
    })
    try:
        matches = buscar_desaparecidos(image_bytes, avistamiento_ref.id)
        warning = None
    except HTTPException as exc:
        matches = []
        warning = exc.detail
    except Exception as exc:
        matches = []
        warning = f"Avistamiento guardado; análisis pendiente: {type(exc).__name__}"
    return {
        "status": "success", "avistamiento_id": avistamiento_ref.id,
        "coincidencia": bool(matches), "coincidencias": matches,
        "total_coincidencias": len(matches), "advertencia": warning,
    }


@app.get("/api/coincidencias", dependencies=[Depends(require_admin_key)])
def listar_coincidencias(folio: Optional[str] = None, avistamiento_id: Optional[str] = None):
    query = db.collection("coincidencias")
    if folio:
        query = query.where("folio", "==", folio)
    if avistamiento_id:
        query = query.where("avistamiento_id", "==", avistamiento_id)
    results = [datos_resultado(doc) for doc in query.stream()]
    results.sort(key=lambda x: x.get("porcentaje_similitud") or 0, reverse=True)
    return {"total": len(results), "coincidencias": results}


@app.get("/api/desaparecidos", dependencies=[Depends(require_admin_key)])
def listar_desaparecidos():
    records = [doc.to_dict() for doc in db.collection("desaparecidos").stream()]
    return {"total": len(records), "desaparecidos": records}


@app.patch("/api/desaparecidos/{folio}/status", dependencies=[Depends(require_admin_key)])
def actualizar_status(folio: str, nuevo_status: str):
    if nuevo_status not in {"activo", "localizado"}:
        raise HTTPException(status_code=422, detail="Estado inválido")
    ref = db.collection("desaparecidos").document(folio)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Folio no encontrado")
    ref.update({"status": nuevo_status})
    db.collection("auditoria").add({"accion": "ACTUALIZACION_STATUS", "folio": folio, "nuevo_status": nuevo_status, "timestamp": firestore.SERVER_TIMESTAMP})
    return {"message": f"Folio {folio} actualizado a {nuevo_status}"}


@app.get("/api/avistamientos", dependencies=[Depends(require_admin_key)])
def listar_avistamientos():
    # Evita enviar todas las fotografías al listar; para imágenes usar coincidencias.
    docs = list(db.collection("avistamientos").stream())
    return {"total": len(docs), "avistamientos": [{"avistamiento_id": d.id} for d in docs]}


@app.post("/api/coincidencias/reanalizar/{folio}", dependencies=[Depends(require_admin_key)])
def reanalizar_desaparecido(folio: str):
    snap = db.collection("desaparecidos").document(folio).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Folio no encontrado")
    encoded = (snap.to_dict() or {}).get("imagen_base64")
    if not encoded:
        raise HTTPException(status_code=409, detail="El registro no tiene fotografía")
    matches = comparar_avistamientos_anteriores(folio, base64.b64decode(encoded))
    return {"folio": folio, "total_coincidencias": len(matches), "coincidencias": matches}


class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/auth/login")
async def login_administrador(credentials: LoginRequest):
    db = firestore.client()
    
    # Buscamos si el correo existe en la colección 'administradores' de Firestore
    admin_ref = db.collection("administradores").where("email", "==", credentials.email).limit(1).get()
    
    if not admin_ref:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El correo no está registrado como administrador."
        )    
    return {
        "success": True,
        "message": "Acceso concedido",
        "email": credentials.email
    }