package ru.alexander.kernel;

import com.aparapi.Kernel;
import org.jcodec.common.model.Picture;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.WritableRaster;
import java.nio.ByteBuffer;

public class RenderKernel extends Kernel {

    private int width;
    private int height;
    public byte[] bytes;

    public double nx;
    public double ny;
    public double nz;
    public double zoom;

    public int len;

    public double[] x;
    public double[] y;
    public double[] z;

    public double[] radius;

    public int[] r;
    public int[] g;
    public int[] b;
    public RenderKernel(Picture shot) {
        bytes = shot.getPlaneData(0);
        width = shot.getWidth();
        height = shot.getHeight();
        setExplicit(true);
    }

    @Override
    public void run() {
        int gid = getGlobalId();
        double cx = (gid % width - width / 2) * ny / zoom;
        double cy = floor(gid / width - height / 2) * nx / zoom;

        double sx = -nz / ny * cx;
        double sy = -nz / nx * cy;
        double sz = cx + cy;
        double maxDst = 0;
        int maxI = -1;
        for (int i = 0; i < len; i++) {
            double t = -(nx * x[i] + ny * y[i] + nz * z[i]) / (nx * nx + ny * ny + nz * nz);
            double px = nx * t + x[i];
            double py = ny * t + y[i];
            double pz = nz * t + z[i];
            double dst = sqrt(pow(sx - px, 2) + pow(sy - py, 2) + pow(sz - pz, 2)) / radius[i];
            if (dst <= 1 && dst >= maxDst) {
                maxDst = dst;
                maxI = i;
            }
        }
        if (maxI != -1) {
            maxDst = cos(1.570796 * maxDst);
            bytes[gid * 3] = (byte) ((int)(maxDst * r[maxI]) - 128);
            bytes[gid * 3 + 1] = (byte) ((int)(maxDst * g[maxI]) - 128);
            bytes[gid * 3 + 2] = (byte) ((int)(maxDst * b[maxI]) - 128);
        }
        else {
            bytes[gid * 3] = -128;
            bytes[gid * 3 + 1] = -128;
            bytes[gid * 3 + 2] = -128;
        }
    }
}
