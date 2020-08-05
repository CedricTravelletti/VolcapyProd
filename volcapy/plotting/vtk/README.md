Try to fully switch to PointClouds (i.e. vtkPolyData structures, to avoid
                                    having to reproject to regular grid).

Point Cloud works. To get nice cells, it then has to go throught the
PointVolumeInterpolator (we use Voronoi and 200x200x30 resolution).
During this process, non-occupied points con give trouble. The strategy it then
to set these to zero.
To do this, one has to deduce bounding box and spacing of the dataset. We do
this by selecting the lowest z-leve (full) and then duplicating this xy slice
over all zlevels in the dataset. We finally do a sett difference to isolate the
non volcano cells and fill these with zeros.

FOR VISUALIZATION: After resampling to image, one should threshold, in order to
get nice cells, otherwise the viz is strange.
