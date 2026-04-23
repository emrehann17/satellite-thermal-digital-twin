import ee
from core.config import GEE_PROJECT

def init_gee(project: str = GEE_PROJECT) -> None:
    ee.Initialize(project=project)

    test_val = ee.Number(42).getInfo()
    if test_val != 42:
        raise RuntimeError("GEE bağlantısı doğrulanamadı.")