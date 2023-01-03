package ru.alexander.kernel;

import com.aparapi.Kernel;

public class MovingKernel extends Kernel {
    public double deltaTime;

    public double[] x;
    public double[] y;
    public double[] z;

    public double[] sx;
    public double[] sy;
    public double[] sz;
    @Override
    public void run() {
        int gid = getGlobalId();
        x[gid] += sx[gid] * deltaTime;
        y[gid] += sy[gid] * deltaTime;
        z[gid] += sz[gid] * deltaTime;
    }
}
