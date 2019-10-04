import ee

# def addTerrain(image):
#
#      terrain = ned.clip(geometry).reproject(image.projection());
#
#      provterrain = terrain.updateMask(image.select(0).mask()).select([0], ['elev']);
#      provaspect = ee.Terrain.aspect(provterrain).select([0], ['aspect']);
#      provslope = ee.Terrain.slope(provterrain).select([0], ['slope']);
#
#     return composite
#         .addBands(provterrain)
#         .addBands(provaspect)
#         .addBands(provslope)


# Add two bands represeting lon/lat of each pixels
def add_latlon(image):

    ll = image.select(0).multiply(0).add(ee.Image.pixelLonLat())

    return image.addBands(ll.select(['longitude', 'latitude'], ['LON', 'LAT']))


def get_checkerboard(image, imgband, updmask, viz, color1, color2):
    # Create a 0/1 checkerboard on a lon/lat grid: take the floor of lon and
    # lat, add them together, and take the low-order bit
    lonlat_checks = ee.Image.pixelLonLat().floor().toInt().reduce(ee.Reducer.sum()).bitwiseAnd(1)

    # Get the image projection from one of the bands
    imgproj = image.select([imgband]).projection()

    # Now replace the projection of the lat/lon checkboard (WGS84 by default)
    # with the desired projection.
    # TODO: it would be a good idea to understand difference between changeProj and reproject.
    imgchecks = lonlat_checks.changeProj(ee.Projection('EPSG:4326'), imgproj)

    # If requested copy the footprint of the image onto the checkerboard,
    # to avoid a global image.
    if updmask:
        imgchecks = imgchecks.updateMask(image.select([imgband]).mask())

    if viz:
        imgchecks = imgchecks.visualize({'min': 0, 'max': 1, 'palette': [color1, color2]})

    return imgchecks


def _rename_band(val, suffix):
    return ee.String(val).cat(ee.String("_")).cat(ee.String(suffix))


def rename_bands(img, suffix):
    bandnames = img.bandNames()
    newnames = bandnames.map(lambda x: _rename_band(x, suffix))
    return img.select(bandnames, newnames)
