from fastapi import APIRouter

router=APIRouter( prefix="/api/v1", tags=["Health Check"])

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "gateway"}


@router.get("/api/v1/hello")
def hello(name: str = "world"):
    return {"message": f"Hello {name} from Gateway!"}