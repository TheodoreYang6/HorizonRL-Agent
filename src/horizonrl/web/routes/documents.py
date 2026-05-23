"""文档管理 API — 上传、列表、删除。"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/api/documents/upload")
async def upload_document(request: Request, file: UploadFile):
    """上传文档: 保存原文件 → 解析文本 → 索引到向量库。"""
    # 校验扩展名
    if file.filename is None:
        return JSONResponse(status_code=400, content={"error": "文件名为空"})
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(status_code=400, content={"error": f"不支持的格式: {suffix}"})

    # 保存文件
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    dest_name = f"{file_id}_{file.filename}"
    dest_path = UPLOAD_DIR / dest_name

    content_bytes = await file.read()
    if len(content_bytes) > MAX_FILE_SIZE:
        return JSONResponse(status_code=400, content={"error": "文件超过 20MB 限制"})

    with open(dest_path, "wb") as f:
        f.write(content_bytes)

    # 解析文本
    try:
        from horizonrl.rag.parser import parse_document
        text = parse_document(str(dest_path), filename=file.filename)
    except ImportError as e:
        (UPLOAD_DIR / dest_name).unlink(missing_ok=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"缺少解析依赖: {e}"},
        )
    except Exception as e:
        (UPLOAD_DIR / dest_name).unlink(missing_ok=True)
        return JSONResponse(status_code=500, content={"error": f"解析失败: {e}"})

    if not text.strip():
        (UPLOAD_DIR / dest_name).unlink(missing_ok=True)
        return JSONResponse(status_code=400, content={"error": "文档内容为空"})

    # 索引
    doc_store = getattr(request.app.state, "document_store", None)
    if doc_store is None:
        return JSONResponse(status_code=500, content={"error": "文档存储未初始化"})

    embedding_client = getattr(request.app.state, "embedding_client", None)

    try:
        result = doc_store.add_document(
            filename=file.filename,
            content=text,
            embedding_client=embedding_client,
        )
    except Exception as e:
        (UPLOAD_DIR / dest_name).unlink(missing_ok=True)
        return JSONResponse(status_code=500, content={"error": f"索引失败: {e}"})

    return {
        "ok": True,
        "doc_id": result["doc_id"],
        "filename": file.filename,
        "chunk_count": result["chunk_count"],
        "total_chars": result["total_chars"],
        "size_bytes": len(content_bytes),
    }


@router.get("/api/documents")
async def list_documents(request: Request):
    """列出已索引的文档。"""
    doc_store = getattr(request.app.state, "document_store", None)
    if doc_store is None:
        return {"documents": []}
    return {"documents": doc_store.list_documents()}


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    """删除指定文档（索引 + 文件）。"""
    doc_store = getattr(request.app.state, "document_store", None)
    if doc_store is not None:
        doc_store.delete_document(doc_id)

    # 删除上传文件
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)

    return {"ok": True, "doc_id": doc_id}
