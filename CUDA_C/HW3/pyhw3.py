import OpenEXR
import Imath
import numpy
import matplotlib.pyplot

pt         = Imath.PixelType(Imath.PixelType.FLOAT)
rgb_hdr    = OpenEXR.InputFile("memorial.exr")
header     = rgb_hdr.header()
dw         = header["dataWindow"]
sz         = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
print header

r_hdr, g_hdr, b_hdr = rgb_hdr.channels("RGB", pt)

r = numpy.fromstring(r_hdr, dtype=numpy.float32).reshape((sz[1], sz[0]))
g = numpy.fromstring(g_hdr, dtype=numpy.float32).reshape((sz[1], sz[0]))
b = numpy.fromstring(b_hdr, dtype=numpy.float32).reshape((sz[1], sz[0]))


red_histo, red_edges = numpy.histogram(r, bins = 275)
print red_histo[0:10]
print sum(red_histo), sum(sum(r)) / r.size
matplotlib.pyplot.plot(red_histo)
matplotlib.pyplot.show()

brightest = max(numpy.max(r), numpy.max(g), numpy.max(b))
print brightest
r, g, b   = r / brightest, g / brightest, b / brightest
rgb       = numpy.concatenate((r[:,:,numpy.newaxis],g[:,:,numpy.newaxis],b[:,:,numpy.newaxis]), axis=2).copy()

#figsize(22, 4)

#matplotlib.pyplot.subplot(161);
#matplotlib.pyplot.imshow(numpy.clip(200 * rgb, 0, 1));
#matplotlib.pyplot.title("numpy.clip(200 * rgb, 0, 1)");

#matplotlib.pyplot.show()
#matplotlib.pyplot.subplot(162);
#matplotlib.pyplot.imshow(numpy.clip(400 * rgb, 0, 1));
#matplotlib.pyplot.title("numpy.clip(400 * rgb, 0, 1)");

#matplotlib.pyplot.subplot(163);
#matplotlib.pyplot.imshow(numpy.clip(800 * rgb, 0, 1));
#matplotlib.pyplot.title("numpy.clip(800 * rgb, 0, 1)");

#matplotlib.pyplot.subplot(164);
#matplotlib.pyplot.imshow(numpy.clip(1600 * rgb, 0, 1));
#matplotlib.pyplot.title("numpy.clip(1600 * rgb, 0, 1)");

#matplotlib.pyplot.subplot(165);
#matplotlib.pyplot.imshow(numpy.clip(3200 * rgb, 0, 1));
#matplotlib.pyplot.title("numpy.clip(3200 * rgb, 0, 1)");

#matplotlib.pyplot.subplot(166);
#matplotlib.pyplot.imshow(numpy.clip(6400 * rgb, 0, 1));
#matplotlib.pyplot.title("numpy.clip(6400 * rgb, 0, 1)");


X = r * 0.4124 + g * 0.3576 + b * 0.1805
Y = r * 0.2126 + g * 0.7152 + b * 0.0722
Z = r * 0.0193 + g * 0.1192 + b * 0.9505

L = X + Y + Z
x = X / L
y = Y / L

delta          = 0.0001

log_Y          = numpy.log10(delta + Y)
min_log_Y      = numpy.min(log_Y)
max_log_Y      = numpy.max(log_Y)
log_Y_range    = max_log_Y - min_log_Y

num_bins       = 1024
hist, edges    = numpy.histogram(log_Y, num_bins)

cdf            = numpy.cumsum(hist)
cdf_total      = cdf[-1]
cdf_norm       = cdf.astype(numpy.float32) / cdf_total

bin_indices    = ( (num_bins * (log_Y - min_log_Y)) / log_Y_range ).astype(numpy.int32) - 1

log_Y_new_norm = cdf_norm[bin_indices]
Y_new          = log_Y_new_norm

X_new = x * (Y_new / y)
Z_new = (1 - x - y) * (Y_new / y);

r_new_cpu = X_new *  3.2406 + Y_new * -1.5372 + Z_new * -0.4986;
g_new_cpu = X_new * -0.9689 + Y_new *  1.8758 + Z_new *  0.0415;
b_new_cpu = X_new *  0.0557 + Y_new * -0.2040 + Z_new *  1.0570;

r_new_cpu = r_new_cpu.astype(numpy.float32)
g_new_cpu = g_new_cpu.astype(numpy.float32)
b_new_cpu = b_new_cpu.astype(numpy.float32)

rgb_new_cpu = numpy.concatenate((r_new_cpu[:,:,numpy.newaxis],g_new_cpu[:,:,numpy.newaxis],b_new_cpu[:,:,numpy.newaxis]), axis=2).copy()

print rgb_new_cpu.shape

data = rgb_new_cpu.tostring()

header['channels']['R'] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
header['channels']['G'] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
header['channels']['B'] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))

ofs = OpenEXR.OutputFile('pythonMemorial.exr', header)
ofs.writePixels({'R': r_new_cpu.tostring(), 'G': g_new_cpu.tostring(), 'B': b_new_cpu.tostring()})


ofs.close()

#for r in range(r_new_cpu.shape[0]):
#    for c in range(r_new_cpu.shape[1]):
#        print r_new_cpu[r][c]


#figsize(14, 9)

#matplotlib.pyplot.subplot(231);
#matplotlib.pyplot.imshow(numpy.clip(1000 * rgb, 0, 1));
#matplotlib.pyplot.title("numpy.clip(1000 * rgb, 0, 1)");

#matplotlib.pyplot.subplot(232);
#matplotlib.pyplot.imshow(Y, cmap="gray");
#matplotlib.pyplot.title("Y");
#matplotlib.pyplot.colorbar();

#matplotlib.pyplot.subplot(233);
#matplotlib.pyplot.imshow(log_Y, cmap="gray");
#matplotlib.pyplot.title("log_Y");
#matplotlib.pyplot.colorbar();


#matplotlib.pyplot.subplot(234);
#matplotlib.pyplot.imshow(bin_indices, cmap="gray");
#matplotlib.pyplot.title("bin_indices");
#matplotlib.pyplot.colorbar();

#matplotlib.pyplot.subplot(235);
#matplotlib.pyplot.imshow(Y_new, cmap="gray");
#matplotlib.pyplot.title("Y_new");
#matplotlib.pyplot.colorbar();

#matplotlib.pyplot.subplot(236);
matplotlib.pyplot.imshow(numpy.clip(rgb_new_cpu, 0, 1));
matplotlib.pyplot.title("numpy.clip(rgb_new_cpu, 0, 1)");
matplotlib.pyplot.show()
