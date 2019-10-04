import ee


def tfexporter(samples, tocloud, selectors, dropselectors, mybucket, prefix, fname):

    if selectors is None:
        selectors = ee.Feature(samples.first()).propertyNames()

    if dropselectors is not None:
        selectors = selectors.removeAll(dropselectors)

    if tocloud:
        task = ee.batch.Export.table.toCloudStorage(

            collection=samples,
            description=fname,
            bucket=mybucket,
            fileNamePrefix=prefix + fname,
            # fileFormat='CSV',
            fileFormat='TFRecord',
            selectors=selectors,
        )

    else:
        task = ee.batch.Export.table.toDrive(

            collection=samples,
            description=fname,
            folder='',
            fileNamePrefix=None,
            # fileFormat= 'CSV',
            fileFormat='TFRecord',
            selectors=selectors

        )

    task.start()

    return task


def _sample_patch(point, patchesarray, scale):
    arrays_samples = patchesarray.sample(
        region=point.geometry(),
        scale=scale,
        #           projection='EPSG:32610',
        projection='EPSG:3857',
        factor=None,
        numPixels=None,
        dropNulls=False,
        tileScale=12

    )

    arrays_samples = ee.Feature(arrays_samples.first())
    return ee.Feature(arrays_samples.copyProperties(point))


def get_array_patches(img, scale, ksize, points, doexport, tocloud,
                      selectors, dropselectors, mybucket, prefix, fname):

    kern = ee.Kernel.square(ksize, 'pixels')
    patches_array = img.neighborhoodToArray(kern)

    # sampleRegions does not cut it for larger collections; using mapped sample instead.
    patches_samps = points.map(lambda pt: _sample_patch(pt, patches_array, scale))  # .flatten();

    if doexport:
        # Export to a TFRecord file in Cloud Storage, creating a file
        # at gs://mybucket/prefix/fname.tfrecord
        # which you can load directly in TensorFlow.
        task = tfexporter(patches_samps, tocloud, selectors, dropselectors, mybucket, prefix, fname)

    return patches_samps


def get_reduced_patches(img, scale, ksize, points, doexport, tocloud,
                        selectors, dropselectors, mybucket, prefix, fname):

    kern = ee.Kernel.square(ksize, 'pixels')
    reducer = ee.Reducer.mean().combine(ee.Reducer.sampleVariance(), "", True)
    patches_reduced = img.reduceNeighborhood(reducer, kern,
                                             inputWeight="kernel",
                                             skipMasked=True,
                                             optimization=None)

    # sampleRegions does not cut it for larger collections; using mapped sample instead.
    patches_samps = points.map(lambda pt: _sample_patch(pt, patches_reduced, scale))  # .flatten();

    if doexport:
        # Export to a TFRecord file in Cloud Storage, creating a file
        # at gs://mybucket/prefix/fname.tfrecord
        # which you can load directly in TensorFlow.
        task = tfexporter(patches_samps, tocloud, selectors, dropselectors, mybucket, prefix, fname)

    return patches_samps