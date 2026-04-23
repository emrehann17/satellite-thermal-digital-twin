import ee

def build_regions() -> dict:
    kozan_merkez = ee.Geometry.Point([35.82, 37.45])
    kozan_aoi = kozan_merkez.buffer(50000).bounds()

    dogu_akdeniz = ee.Geometry.BBox(33.8, 36.0, 36.7, 38.0)

    return {
        "dogu_akdeniz": dogu_akdeniz,
        "kozan_aoi": kozan_aoi
    }