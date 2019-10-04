"""
Author: George Azzari (gazzari@stanford.edu)
Center on Food Security and the Environment
Department of Earth System Science
Stanford University
"""

import ee


def _mergejoin(joinedelement):
    #  Inner join returns a FeatureCollection with a primary and secondary set of
    #  properties. Properties are collapsed into different bands of an image.
    return ee.Image.cat(joinedelement.get('primary'), joinedelement.get('secondary'))


#  Convenience function for joining two collections based on system:time_start
def joincoll(coll1, coll2):
    eqfilter = ee.Filter.equals(rightField='system:time_start', leftField='system:time_start')
    join = ee.Join.inner()
    joined = ee.ImageCollection(join.apply(coll1, coll2, eqfilter))
    return joined.map(_mergejoin).sort('system:time_start')


def addVIs(img):

    evi = img.expression(
        '2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)',
        {'red': img.select('RED'),
         'nir': img.select('NIR'),
         'blue': img.select('BLUE')
         }).select([0], ['EVI'])

    gcvi = img.expression(
        '(nir / green) - 1',
        {'nir': img.select('NIR'),
         'green': img.select('GREEN')
         }).select([0], ['GCVI'])

    tvi = img.expression(
        '0.5 * (120 * (nir - green) - 200 * (red - green))',
        {'nir': img.select('NIR'),
         'green': img.select('GREEN'),
         'red': img.select('RED')
         }).select([0], ['TVI'])

    sndvi = img.expression(
        '(nir - red) / (red + nir + 0.16)',
        {'nir': img.select('NIR'),
         'red': img.select('RED')
         }).select([0], ['SNDVI'])

    ndvi = img.expression(
        '(nir - red) / (red + nir)',
        {'nir': img.select('NIR'),
         'red': img.select('RED')
         }).select([0], ['NDVI'])

    nbr1 = img.expression(
        '(nir - swir1) / (nir + swir1)',
        {'nir': img.select('NIR'),
         'swir1': img.select('SWIR1')
         }).select([0], ['NBR1'])

    nbr2 = img.expression(
        '(nir - swir2) / (nir + swir2)',
        {'nir': img.select('NIR'),
         'swir2': img.select('SWIR2')
         }).select([0], ['NBR2'])

    # Simple tillage index
    sti = img.expression(
        'swir1/swir2',
        {'swir1': img.select('SWIR1'),
         'swir2': img.select('SWIR2')
         }).select([0], ['STI'])

    # NDTI
    ndti = img.expression(
        '(swir1 - swir2) / (swir1 + swir2)',
        {'swir1': img.select('SWIR1'),
         'swir2': img.select('SWIR2')
         }).select([0], ['NDTI'])

    # Modified CRC
    crc = img.expression(
        '(swir1 - green) / (swir1 + green)',
        {'green': img.select('GREEN'),
         'swir1': img.select('SWIR1')
         }).select([0], ['CRC'])

    return ee.Image.cat([img, evi, gcvi, tvi, sndvi, ndvi, nbr1, nbr2, sti, ndti, crc])


# TODO: add Landsat TOA (c01)

class LandsatTOAPRE:

    def __init__(self, filterpoly, start_date, end_date):

        self.filterpoly = filterpoly

        self.s = start_date
        self.e = end_date

        self.l8 = self._init_coll('LANDSAT/LC8_L1T_TOA_FMASK')
        self.l7 = self._init_coll('LANDSAT/LE7_L1T_TOA_FMASK')
        self.l5 = self._init_coll('LANDSAT/LT5_L1T_TOA_FMASK')

        self.l8fm = self.l8.map(self._mask_fm).map(self._rename_l8)
        self.l7fm = self.l7.map(self._mask_fm).map(self._rename_l7)
        self.l5fm = self.l5.map(self._mask_fm).map(self._rename_l5)

        self.merged = ee.ImageCollection(self.l5.merge(self.l7).merge(self.l8)).sort('system:time_start')
        self.mergedfm = ee.ImageCollection(self.l5fm.merge(self.l7fm).merge(self.l8fm)).sort('system:time_start')

    @staticmethod
    def _pansharpen(image):

        # Convert the RGB bands to the HSV color space.
        rgb = image.select('RED', 'BLUE', 'GREEN')
        gray = image.select('PAN')

        # Swap in the panchromatic band and convert back to RGB.
        hsv = rgb.rgbToHsv().select('hue', 'saturation')

        rgb_sharpened = ee.Image.cat(hsv, gray).hsvToRgb()

        return rgb_sharpened

    def _init_coll(self, name):

        return ee.ImageCollection(name).filterBounds(self.filterpoly).filterDate(self.s, self.e)

    @staticmethod
    def _mask_fm(img):

        """
        fmask: cloud
        mask
        0 = clear
        1 = water
        2 = shadow
        3 = snow
        4 = cloud
        """

        fmask = img.select('fmask')
        # goodpx = fmask.lt(2)
        goodpx = fmask.eq(0)

        return img.updateMask(goodpx).select(img.bandNames().removeAll(['fmask', 'BQA']))

    @staticmethod
    def _rescale_all_bands(img, reflbands, thermbands):

        reflimg = img.select(reflbands).multiply(1000).toInt16()
        thermimg = img.select(thermbands).multiply(10).toInt16()

        return reflimg.addBands(thermimg)

    @staticmethod
    def _rename_l8(l8img):

        """
        Bands of Landsat 8 are:
        B1: Coastal aerosol (0.43 - 0.45 um)
        B2: Blue (0.45 - 0.51 um)
        B3: Green (0.53 - 0.59 um)
        B4: Red (0.64 - 0.67 um)
        B5: Near Infrared (0.85 - 0.88 um)
        B6: Short-wave Infrared 1 (1.57 - 1.65 um)
        B7: Short-wave infrared 2 (2.11 - 2.29 um)
        B8: Panchromatic (0.50 - 0.68 um)
        B9: Cirrus (1.36 - 1.38 um)
        B10: Thermal Infrared 1 (10.60 - 11.19 um)
        B11: Thermal Infrared 2 (11.50 - 12.51 um)
        :param l8img:
        :return:
        """

        bands = ['AEROS', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'PAN', 'CIRR', 'THER1', 'THER2']
        reflimg = l8img.select([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bands)

        return reflimg

    @staticmethod
    def _rename_l7(l7img):

        """
        B1 - blue	0.45 - 0.52
        B2 - green	0.52 - 0.60
        B3 - red	0.63 - 0.69
        B4 - Near Infrared	0.77 - 0.90
        B5 - Short-wave Infrared	1.55 - 1.75
        B6_VCID_1 - Thermal Infrared	10.40 - 12.50
        B6_VCID_2 - Thermal Infrared	10.40 - 12.50
        B7 - Short-wave Infrared	2.09 - 2.35
        B8 - Panchromatic (Landsat 7 only)	0.52 - 0.90
        """

        reflbands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'PAN']
        reflimg = l7img.select([0, 1, 2, 3, 4, 7, 8], reflbands)

        thermbands = ['THER1', 'THER2']
        thermimg = l7img.select([5, 6], thermbands)

        return reflimg.addBands(thermimg)

    @staticmethod
    def _rename_l5(l5img):

        """
        B1 - blue	0.45 - 0.52
        B2 - green	0.52 - 0.60
        B3 - red	0.63 - 0.69
        B4 - Near Infrared	0.77 - 0.90
        B5 - Short-wave Infrared	1.55 - 1.75
        B6 - Thermal Infrared	10.40 - 12.50
        B7 - Short-wave Infrared	2.09 - 2.35
        """

        reflbands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
        reflimg = l5img.select([0, 1, 2, 3, 4, 6], reflbands)

        thermbands = ['THER1']
        thermimg = l5img.select([5], thermbands)

        return reflimg.addBands(thermimg)


class LandsatSRPRE:
    def __init__(self, filterpoly, start_date, end_date):
        self.filterpoly = filterpoly
        self.s = start_date
        self.e = end_date
        self.l8 = self.init_coll8('LANDSAT/LC8_SR').map(self.fixwrs)
        self.l7 = self.init_coll('LANDSAT/LE7_SR').map(self.fixwrs)
        self.l5 = self.init_coll('LANDSAT/LT5_SR').map(self.fixwrs)
        self.l8cfm = self.l8.map(self.cfmask)
        self.l7cfm = self.l7.map(self.cfmask)
        self.l5cfm = self.l5.map(self.cfmask)
        self.l7qam = self.l7.map(self.mask_qa)
        self.l5qam = self.l5.map(self.mask_qa)
        # todo: update names to actual colors
        self.merged = ee.ImageCollection(self.l5.merge(self.l7).merge(self.l8)).sort('system:time_start')
        self.mergedcfm = ee.ImageCollection(self.l5cfm.merge(self.l7cfm).merge(self.l8cfm)).sort('system:time_start')
        self.mergedqam = ee.ImageCollection(self.l5qam.merge(self.l7qam)).sort('system:time_start')

    def init_coll(self, name):
        return ee.ImageCollection(name).filterBounds(self.filterpoly).filterDate(self.s, self.e).map(self.rename_l457)

    def init_coll8(self, name):
        return ee.ImageCollection(name).filterBounds(self.filterpoly).filterDate(self.s, self.e).map(self.rename_l8)

    @staticmethod
    def fixwrs(srimg):
        return srimg.set({'WRS_PATH': srimg.get('wrs_path'), 'WRS_ROW': srimg.get('wrs_row')})

    @staticmethod
    def rename_l8(l8img):
        """
        Bands of Landsat 8 are:
        B1: Coastal aerosol (0.43 - 0.45 um) (signed int16)
        B2: Blue (0.45 - 0.51 um) (signed int16)
        B3: Green (0.53 - 0.59 um) (signed int16)
        B4: Red (0.64 - 0.67 um) (signed int16)
        B5: Near Infrared (0.85 - 0.88 um) (signed int16)
        B6: Short-wave Infrared 1 (1.57 - 1.65 um) (signed int16)
        B7: Short-wave infrared 2 (2.11 - 2.29 um) (signed int16)
        cfmask - cloud mask (unsigned int8)
        cfmask_conf - cloud mask confidence (unsigned int8)
        :param l8img:
        :return:
        """
        return l8img.rename(['AEROS', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'cfmask', 'cfmask_conf'])

    @staticmethod
    def rename_l457(limg):
        """
        B1 - blue	0.45 - 0.52
        B1 - green	0.52 - 0.60
        B3 - red	0.63 - 0.69
        B4 - Near Infrared	0.77 - 0.90
        B5 - Short-wave Infrared	1.55 - 1.75
        B7 - Short-wave Infrared	2.09 - 2.35
        cfmask - cloud mask
        cfmask_conf - cloud mask confidence
        adjacent_cloud_qa - Binary QA mask (0/255)
        cloud_qa - Binary QA mask (0/255)
        cloud_shadow_qa - Binary QA mask (0/255)
        ddv_qa, fill_qa - Binary QA mask (0/255)
        land_water_qa - Binary QA mask (0/255)
        snow_qa - Binary QA mask (0/255)
        atmos_opacity - Atmospheric Opacity
        """
        return limg.rename(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'cfmask', 'cfmask_conf',
                            'adjacent_cloud_qa', 'cloud_qa', 'cloud_shadow_qa', 'ddv_qa', 'fill_qa',
                            'land_water_qa', 'snow_qa', 'atmos_opacity'])

    @staticmethod
    def mask_qa(srimg):
        """
        QA bands are only available for Landsat 4-7:
        adjacent_cloud_qa,
        cloud_qa,
        cloud_shadow_qa,
        ddv_qa, (dense dark vegetation)
        fill_qa,
        land_water_qa,
        snow_qa
        255 if the corresponding condition was detected
        0 otherwise
        :param srimg: SR Landsat image
        :return: SR Landsat image masked for clear pixels only
        """
        qabands = ['cloud_qa', 'cloud_shadow_qa', 'adjacent_cloud_qa', 'land_water_qa', 'snow_qa']
        badqamask = srimg.select(qabands).reduce(ee.Reducer.anyNonZero())
        totqamask = ee.Image(1).mask(srimg.select('BLUE').mask()).where(badqamask, 0)
        return srimg.updateMask(totqamask)

    @staticmethod
    def cfmask(srimg):
        """
        cfmask: cloud mask.
        0=clear
        1=water
        2=shadow
        3=snow
        4=cloud
        cfmask_conf: cloud mask confidence
        0=none
        1=cloud confidence >= 12.5
        2=cloud confidence > 12.5% and <= 22.5%
        3=cloud confidence > 22.5
        :param srimg:
        :return:
        """
        cfmask = srimg.select('cfmask')
        clearpx = cfmask.eq(0)
        return srimg.updateMask(clearpx)


class LandsatSR:
    """
    Image Properties

    Name	                Definition
    CLOUD_COVER	            Percentage cloud cover, -1 = not calculated
    CLOUD_COVER_LAND	    Percentage cloud cover over land, -1 = not calculated
    IMAGE_QUALITY	        Image quality, 0 = worst, 9 = best, -1 = quality not calculated
    EARTH_SUN_DISTANCE	    Earth-Sun distance (AU)
    ESPA_VERSION	        Internal ESPA image version used to process SR
    LANDSAT_ID	            Landsat Product Identifier (Collection 1)
    LEVEL1_PRODUCTION_DATE	Date of production for raw level 1 data
    PIXEL_QA_VERSION	    Version of the software used to produce the pixel_qa band
    SATELLITE	            Name of satellite
    SOLAR_AZIMUTH_ANGLE	    Solar azimuth angle
    SR_APP_VERSION	        LaSRC version used to process surface reflectance
    WRS_PATH	            WRS-2 path number of scene
    WRS_ROW	                WRS row number of scene
    """
    def __init__(self, filterpoly, start_date, end_date):

        self.filterpoly = filterpoly
        self.s = start_date
        self.e = end_date

        self.l8 = self.init_coll8('LANDSAT/LC08/C01/T1_SR').map(self.rename_l8).map(self.rescale_l8)
        self.l7 = self.init_coll('LANDSAT/LE07/C01/T1_SR').map(self.rename_l57).map(self.rescale_l57)
        self.l5 = self.init_coll('LANDSAT/LT05/C01/T1_SR').map(self.rename_l57).map(self.rescale_l57)

        # There are other ways of masking, this is just the simpler and more commonly used.
        self.l8qam = self.l8.map(self.mask_qaclear_l8)
        self.l7qam = self.l7.map(self.mask_qaclear_l57)
        self.l5qam = self.l5.map(self.mask_qaclear_l57)

        # Merging some of the collections that more commonly used
        self.merged = ee.ImageCollection(self.l5.merge(self.l7).merge(self.l8)).sort('system:time_start')
        self.mergedqam = ee.ImageCollection(self.l5qam.merge(self.l7qam).merge(self.l8qam)).sort('system:time_start')

    def init_coll(self, name):
        return ee.ImageCollection(name).filterBounds(self.filterpoly).filterDate(self.s, self.e).map(self.rename_l57)

    def init_coll8(self, name):
        return ee.ImageCollection(name).filterBounds(self.filterpoly).filterDate(self.s, self.e).map(self.rename_l8)

    @staticmethod
    def rename_l8(l8img):
        """
        Name	Scale Factor	Description
        B1	    0.0001	    Band 1 (Ultra Blue) surface reflectance, 0.435-0.451 um
        B2	    0.0001	    Band 2 (Blue) surface reflectance, 0.452-0.512 um
        B3	    0.0001	    Band 3 (Green) surface reflectance, 0.533-0.590 um
        B4	    0.0001	    Band 4 (Red) surface reflectance, 0.636-0.673 um
        B5	    0.0001	    Band 5 (Near Infrared) surface reflectance, 0.851-0.879 um
        B6	    0.0001	    Band 6 (Shortwave Infrared 1) surface reflectance, 1.566-1.651 um
        B7	    0.0001	    Band 7 (Shortwave Infrared 2) surface reflectance, 2.107-2.294 um
        B10	    0.1	        Band 10 brightness temperature (Kelvin), 10.60-11.19 um
        B11	    0.1	        Band 11 brightness temperature (Kelvin), 11.50-12.51 um
        sr_aerosol		    Aerosol attributes, see Aerosol QA table
        pixel_qa		    Pixel quality attributes, see Pixel QA table
        radsat_qa		    Radiometric saturation QA, see Radsat QA table
        """

        newnames = ['AEROS', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2',
                    'TEMP1', 'TEMP2', 'sr_aerosol', 'pixel_qa', 'radsat_qa']

        return l8img.rename(newnames)

    @staticmethod
    def rescale_l8(scene):

        opt = scene.select(['AEROS', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
        therm = scene.select(['TEMP1', 'TEMP2'])
        masks = scene.select(['sr_aerosol', 'pixel_qa', 'radsat_qa'])

        opt = opt.multiply(0.0001)
        therm = therm.multiply(0.1)

        scaled = ee.Image(ee.Image.cat([opt, therm, masks]).copyProperties(scene))
        # System properties are not copied (?)
        scaled = scaled.set('system:time_start', scene.get('system:time_start'))

        return scaled

    @staticmethod
    def rename_l57(limg):
        """
        Name	Scale Factor	Description
        B1	    0.0001	    Band 1 (blue) surface reflectance, 0.45-0.52 um
        B2	    0.0001	    Band 2 (green) surface reflectance, 0.52-0.60 um
        B3	    0.0001	    Band 3 (red) surface reflectance, 0.63-0.69 um
        B4	    0.0001	    Band 4 (near infrared) surface reflectance, 0.77-0.90 um
        B5	    0.0001	    Band 5 (shortwave infrared 1) surface reflectance, 1.55-1.75 um
        B6	    0.1	        Band 6 brightness temperature (Kelvin), 10.40-12.50 um
        B7	    0.0001	    Band 7 (shortwave infrared 2) surface reflectance, 2.08-2.35 um
        sr_atmos_opacity	0.001	Atmospheric opacity; < 0.1 = clear; 0.1 - 0.3 = average; > 0.3 = hazy
        sr_cloud_qa		    Cloud quality attributes, see SR Cloud QA table. Note:
                            pixel_qa is likely to present more accurate results
                            than sr_cloud_qa for cloud masking. See page 23 in
                            the LEDAPS product guide.
        pixel_qa		    Pixel quality attributes generated from the CFMASK algorithm,
                            see Pixel QA table
        radsat_qa		Radiometric saturation QA, see Radiometric Saturation QA table
        """
        return limg.rename(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'TEMP1', 'SWIR2',
                            'sr_atmos_opacity', 'sr_cloud_qa', 'pixel_qa', 'radsat_qa'])

    @staticmethod
    def rescale_l57(scene):

        opt = scene.select(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
        atmos = scene.select(['sr_atmos_opacity'])
        therm = scene.select(['TEMP1'])
        masks = scene.select(['sr_cloud_qa', 'pixel_qa', 'radsat_qa'])

        opt = opt.multiply(0.0001)
        atmos = atmos.multiply(0.001)
        therm = therm.multiply(0.1)

        scaled = ee.Image(ee.Image.cat([opt, therm, masks, atmos]).copyProperties(scene))
        # System properties are not copied (?)
        scaled = scaled.set('system:time_start', scene.get('system:time_start'))

        return scaled


    @staticmethod
    def decode_qamask_l8(scene):
        """
        Pixel QA Bit Flags
        Bit	Attribute
        0	Fill
        1	Clear
        2	Water
        3	Cloud Shadow
        4	Snow
        5	Cloud
        6-7	Cloud Confidence (00 = None, 01 = Low, 10 = Medium, 11 = High)
        8-9	Cirrus Confidence (00 = None, 01 = Low, 0 = Medium, 11 = High)
        10	Terrain Occlusion
        """

        qa = scene.select('pixel_qa')
        clear = qa.bitwiseAnd(2).neq(0)
        clear = clear.updateMask(clear).rename(['pxqa_clear'])

        water = qa.bitwiseAnd(4).neq(0)
        water = water.updateMask(water).rename(['pxqa_water'])

        cloud_shadow = qa.bitwiseAnd(8).neq(0)
        cloud_shadow = cloud_shadow.updateMask(cloud_shadow).rename(['pxqa_cloudshadow'])

        snow = qa.bitwiseAnd(16).neq(0)
        snow = snow.updateMask(snow).rename(['pxqa_snow'])

        cloud = qa.bitwiseAnd(32).neq(0)
        cloud = cloud.updateMask(cloud).rename(['pxqa_cloud'])

        # Cloud confidence is comprised of bits 6-7.
        # Add the two bits and interpolate them to a range from 0-3.
        # 0 = None, 1 = Low, 2 = Medium, 3 = High.
        cloud_conf = qa.bitwiseAnd(64).add(qa.bitwiseAnd(128)).interpolate([0, 64, 128, 192], [0, 1, 2, 3], 'clamp')
        cloud_conf = cloud_conf.int().rename(['pxqa_cloudconf'])

        # Cirrus confidence is comprised of bits 8-9.
        # Add the two bits and interpolate them to a range from 0-3.
        # 0 = None, 1 = Low, 2 = Medium, 3 = High.
        cirrus_conf = qa.bitwiseAnd(256).add(qa.bitwiseAnd(512)).interpolate([0, 64, 128, 192], [0, 1, 2, 3], 'clamp')
        cirrus_conf = cirrus_conf.int().rename(['pxqa_cirrusconf'])

        terrain = qa.bitwiseAnd(1024).neq(0)
        terrain = terrain.updateMask(terrain).rename(['qa_terrain'])

        masks = ee.Image.cat([
            clear, water, cloud_shadow, snow,
            cloud, cloud_conf, cirrus_conf,
            terrain
        ])

        # return scene.select(scene.bandNames().remove('pixel_qa')).addBands(masks)
        return masks

    @staticmethod
    def decode_qamask_l57(scene):
        """
        Pixel QA Bit Flags
        Bit	Attribute
        0	Fill
        1	Clear
        2	Water
        3	Cloud Shadow
        4	Snow
        5	Cloud
        6-7	Cloud Confidence (00 = None, 01 = Low, 10 = Medium, 11 = High)
        """

        qa = scene.select('pixel_qa')
        clear = qa.bitwiseAnd(2).neq(0)
        clear = clear.updateMask(clear).rename(['pxqa_clear'])

        water = qa.bitwiseAnd(4).neq(0)
        water = water.updateMask(water).rename(['pxqa_water'])

        cloud_shadow = qa.bitwiseAnd(8).neq(0)
        cloud_shadow = cloud_shadow.updateMask(cloud_shadow).rename(['pxqa_cloudshadow'])

        snow = qa.bitwiseAnd(16).neq(0)
        snow = snow.updateMask(snow).rename(['pxqa_snow'])

        cloud = qa.bitwiseAnd(32).neq(0)
        cloud = cloud.updateMask(cloud).rename(['pxqa_cloud'])

        # Cloud confidence is comprised of bits 6-7.
        # Add the two bits and interpolate them to a range from 0-3.
        # 0 = None, 1 = Low, 2 = Medium, 3 = High.
        cloud_conf = qa.bitwiseAnd(64).add(qa.bitwiseAnd(128)).interpolate([0, 64, 128, 192], [0, 1, 2, 3], 'clamp')
        cloud_conf = cloud_conf.int().rename(['pxqa_cloudconf'])

        masks = ee.Image.cat([
            clear, water, cloud_shadow, snow,
            cloud, cloud_conf
        ])

        # return scene.select(scene.bandNames().remove('pixel_qa')).addBands(masks)
        return masks

    def mask_qaclear_l8(self, scene):
        clearmask = self.decode_qamask_l8(scene).select('pxqa_clear')
        return scene.updateMask(clearmask)

    def mask_qaclear_l57(self, scene):
        clearmask = self.decode_qamask_l57(scene).select('pxqa_clear')
        return scene.updateMask(clearmask)


class LandsatJoined:
    def __init__(self, filterpoly, start_date, end_date):
        self.toa = LandsatTOAPRE(filterpoly, start_date, end_date)
        self.sr = LandsatSRPRE(filterpoly, start_date, end_date)
        # TODO: does doing three joins separately cost more than a single all-inclusive join?
        self.l8 = joincoll(self.sr.l8sel, self.toa.l8sel).map(self._cast_img2float)
        self.l7 = joincoll(self.sr.l7sel, self.toa.l7sel).map(self._cast_img2float)
        self.l5 = joincoll(self.sr.l5sel, self.toa.l5sel).map(self._cast_img2float)
        self.merged = ee.ImageCollection(self.l5.merge(self.l7).merge(self.l8))
        self.merged = self.merged.sort('system:time_start')

    @staticmethod
    def _cast_img2float(img):
        return img.toFloat()


class MODISrefl(object):
    def __init__(self, start_date, end_date):
        self.newnames = ['RED', 'NIR', 'BLUE', 'GREEN', 'SWIR1', 'SWIR2']
        self.orignames = None
        self.collname = None
        self.coll = None
        self.start_date = start_date
        self.end_date = end_date

    def rename(self, img):
        return img.select(self.orignames, self.newnames)


class MODISnbar(MODISrefl):
    """
    MODIS BRDF-adjusted Reflectance 16-day Global 500m
    """
    def __init__(self, start_date, end_date):
        MODISrefl.__init__(self, start_date, end_date)
        self.orignames = ['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band3',
                          'Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band6', 'Nadir_Reflectance_Band7']
        self.collname = "MODIS/MCD43A4"
        self.coll = ee.ImageCollection(self.collname).filterDate(start_date, end_date).map(self.rename)


class MODISsr(MODISrefl):
    """
    MODIS Surface Reflectance 8-Day Global 500m
    """
    def __init__(self, start_date, end_date):
        MODISrefl.__init__(self, start_date, end_date)
        self.orignames = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03',
                          'sur_refl_b04', 'sur_refl_b06', 'sur_refl_b07']
        self.collname = "MODIS/MOD09A1"
        self.coll = ee.ImageCollection(self.collname).filterDate(start_date, end_date).map(self.rename)


class Sentinel2TOA(object):

    def __init__(self, filterpoly, start_date, end_date):

        self.filterpoly = filterpoly

        self.s = start_date

        self.e = end_date

    def init_coll(self, name):

        return ee.ImageCollection(name).filterBounds(self.filterpoly).filterDate(self.s, self.e).map(self.rename)

    @staticmethod
    def qa_cloudmask(img):

        # Opaque and cirrus cloud masks cause bits 10 and 11 in QA60 to be set,so values less than 1024 are cloud-free
        mask = ee.Image(0).where(img.select('QA60').gte(1024), 1).Not()

        return img.updateMask(mask)

    @staticmethod
    def _rescale(img, exp, thresholds):

        """
        A helper to apply an expression and linearly rescale the output.
        """

        return img.expression(exp, img=img).subtract(thresholds[0]).divide(thresholds[1] - thresholds[0])

    def add_cloud_score(self, img):

        img = ee.Image(img).divide(1000)

        score = ee.Image(1.0)

        score = score.min(self._rescale(img, 'img.cirrus', [0, 0.1]))

        score = score.min(self._rescale(img, 'img.cb', [0.5, 0.8]))

        score = score.min(self._rescale(img.normalizedDifference(['GREEN', 'SWIR1']), 'img', [0.8, 0.6]))

        # Invert the cloudscore so 1 is least cloudy, and rename the band.
        return img.addBands(ee.Image(1).subtract(score).select([0], ['cloudscore']))

    @staticmethod
    def rename(s2img):

        """
        Band	Use	Wavelength	Resolution
        B1	Aerosols	443nm	60m
        B2	Blue	490nm	10m
        B3	Green	560nm	10m
        B4	Red	665nm	10m
        B5	Red Edge 1	705nm	20m
        B6	Red Edge 2	740nm	20m
        B7	Red Edge 3	783nm	20m
        B8	NIR	842nm	10m
        B8a	Red Edge 4	865nm	20m
        B9	Water vapor	940nm	60m
        B10	Cirrus	1375nm	60m
        B11	SWIR 1	1610nm	20m
        B12	SWIR 2	2190nm	20m
        QA10
        QA20
        QA60
        """

        newnames = ['AEROS', 'BLUE', 'GREEN', 'RED', 'RDED1', 'RDED2', 'RDED3',
                    'NIR', 'RDED4', 'VAPOR', 'CIRRU', 'SWIR1', 'SWIR2', 'QA10',
                    'QA20', 'QA60']

        return s2img.rename(newnames)

