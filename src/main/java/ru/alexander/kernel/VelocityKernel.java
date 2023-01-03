package ru.alexander.kernel;

import com.aparapi.Kernel;

public class VelocityKernel extends Kernel {
    public int len;

    public double G;
    public double deltaTime;

    public double[] mass;

    public double[] x;
    public double[] y;
    public double[] z;

    public double[] sx;
    public double[] sy;
    public double[] sz;

    public double[] error;

    @Override
    public void run() {
        int gid = getGlobalId();

        double[] k1 = new double[3];
        k1[0] = 0;
        k1[1] = 0;
        k1[2] = 0;
        for (int i = 0; i < len; i++)
            if (gid != i) {
                double dx = x[i] - x[gid];
                double dy = y[i] - y[gid];
                double dz = z[i] - z[gid];
                double sqr = dx * dx + dy * dy + dz * dz;
                double v = G * mass[i] * deltaTime / sqr / sqrt(sqr);
                k1[0] += v * dx;
                k1[1] += v * dy;
                k1[2] += v * dz;
            }

        double[] k2 = new double[3];
        k2[0] = 0;
        k2[1] = 0;
        k2[2] = 0;
        for (int i = 0; i < len; i++)
            if (gid != i) {
                double dx = x[i] - (x[gid] + k1[0] / 5);
                double dy = y[i] - (y[gid] + k1[1] / 5);
                double dz = z[i] - (z[gid] + k1[2] / 5);
                double sqr = dx * dx + dy * dy + dz * dz;
                double v = G * mass[i] * deltaTime / sqr / sqrt(sqr);
                k2[0] += v * dx;
                k2[1] += v * dy;
                k2[2] += v * dz;
            }

        double[] k3 = new double[3];
        k3[0] = 0;
        k3[1] = 0;
        k3[2] = 0;
        for (int i = 0; i < len; i++)
            if (gid != i) {
                double dx = x[i] - (x[gid] + (k1[0] * 3 + k2[0] * 9) / 40);
                double dy = y[i] - (y[gid] + (k1[1] * 3 + k2[1] * 9) / 40);
                double dz = z[i] - (z[gid] + (k1[2] * 3 + k2[2] * 9) / 40);
                double sqr = dx * dx + dy * dy + dz * dz;
                double v = G * mass[i] * deltaTime / sqr / sqrt(sqr);
                k3[0] += v * dx;
                k3[1] += v * dy;
                k3[2] += v * dz;
            }

        double[] k4 = new double[3];
        k4[0] = 0;
        k4[1] = 0;
        k4[2] = 0;
        for (int i = 0; i < len; i++)
            if (gid != i) {
                double dx = x[i] - (x[gid] + (k1[0] * 44 + k3[0] * 160 - k2[0] * 168) / 45);
                double dy = y[i] - (y[gid] + (k1[1] * 44 + k3[1] * 160 - k2[1] * 168) / 45);
                double dz = z[i] - (z[gid] + (k1[2] * 44 + k3[2] * 160 - k2[2] * 168) / 45);
                double sqr = dx * dx + dy * dy + dz * dz;
                double v = G * mass[i] * deltaTime / sqr / sqrt(sqr);
                k4[0] += v * dx;
                k4[1] += v * dy;
                k4[2] += v * dz;
            }

        double[] k5 = new double[3];
        k5[0] = 0;
        k5[1] = 0;
        k5[2] = 0;
        for (int i = 0; i < len; i++)
            if (gid != i) {
                double dx = x[i] - (x[gid] + (k1[0] * 19372 + k3[0] * 64448 - k2[0] * 76080 - k4[0] * 1908) / 6561);
                double dy = y[i] - (y[gid] + (k1[1] * 19372 + k3[1] * 64448 - k2[1] * 76080 - k4[1] * 1908) / 6561);
                double dz = z[i] - (z[gid] + (k1[2] * 19372 + k3[2] * 64448 - k2[2] * 76080 - k4[2] * 1908) / 6561);
                double sqr = dx * dx + dy * dy + dz * dz;
                double v = G * mass[i] * deltaTime / sqr / sqrt(sqr);
                k5[0] += v * dx;
                k5[1] += v * dy;
                k5[2] += v * dz;
            }

        double[] k6 = new double[3];
        k6[0] = 0;
        k6[1] = 0;
        k6[2] = 0;
        for (int i = 0; i < len; i++)
            if (gid != i) {
                double dx = x[i] - (x[gid] + k1[0] * 9017.0 / 3168 + k3[0] * 46732.0 / 5247 + k4[0] * 49.0 / 176 - k2[0] * 355.0 / 33 - k5[0] * 5103.0 / 18656);
                double dy = y[i] - (y[gid] + k1[1] * 9017.0 / 3168 + k3[1] * 46732.0 / 5247 + k4[1] * 49.0 / 176 - k2[1] * 355.0 / 33 - k5[1] * 5103.0 / 18656);
                double dz = z[i] - (z[gid] + k1[2] * 9017.0 / 3168 + k3[2] * 46732.0 / 5247 + k4[2] * 49.0 / 176 - k2[2] * 355.0 / 33 - k5[2] * 5103.0 / 18656);
                double sqr = dx * dx + dy * dy + dz * dz;
                double v = G * mass[i] * deltaTime / sqr / sqrt(sqr);
                k6[0] += v * dx;
                k6[1] += v * dy;
                k6[2] += v * dz;
            }

        double vx = k1[0] * 35.0 / 384 + k3[0] * 500.0 / 1113 + k4[0] * 125.0 / 192 + k6[0] * 11.0 / 84 - k5[0] * 2187.0 / 6784;
        double vy = k1[1] * 35.0 / 384 + k3[1] * 500.0 / 1113 + k4[1] * 125.0 / 192 + k6[1] * 11.0 / 84 - k5[1] * 2187.0 / 6784;
        double vz = k1[2] * 35.0 / 384 + k3[2] * 500.0 / 1113 + k4[2] * 125.0 / 192 + k6[2] * 11.0 / 84 - k5[2] * 2187.0 / 6784;

        double[] k7 = new double[3];
        k7[0] = 0;
        k7[1] = 0;
        k7[2] = 0;
        for (int i = 0; i < len; i++)
            if (gid != i) {
                double dx = x[i] - (x[gid] + vx);
                double dy = y[i] - (y[gid] + vy);
                double dz = z[i] - (z[gid] + vz);
                double sqr = dx * dx + dy * dy + dz * dz;
                double v = G * mass[i] * deltaTime / sqr / sqrt(sqr);
                k7[0] += v * dx;
                k7[1] += v * dy;
                k7[2] += v * dz;
            }

        double tx = (k1[0] * 5179.0 / 90 + k4[0] * 393.0) / 640 + k3[0] * 7571.0 / 16695 + k6[0] * 187.0 / 2100 + k7[0] / 40 - k5[0] * 92097.0 / 339200;
        double ty = (k1[1] * 5179.0 / 90 + k4[1] * 393.0) / 640 + k3[1] * 7571.0 / 16695 + k6[1] * 187.0 / 2100 + k7[1] / 40 - k5[1] * 92097.0 / 339200;
        double tz = (k1[2] * 5179.0 / 90 + k4[2] * 393.0) / 640 + k3[2] * 7571.0 / 16695 + k6[2] * 187.0 / 2100 + k7[2] / 40 - k5[2] * 92097.0 / 339200;

        sx[gid] += vx;
        sy[gid] += vy;
        sz[gid] += vz;

        error[gid] = sqrt(pow(vx - tx, 2) + pow(vy - ty, 2) + pow(vz - tz, 2));
    }


}
