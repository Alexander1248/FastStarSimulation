package ru.alexander.objects;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.exception.CompileFailedException;
import com.aparapi.internal.kernel.KernelManager;
import org.jcodec.api.SequenceEncoder;
import org.jcodec.common.model.ColorSpace;
import org.jcodec.common.model.Picture;
import org.jcodec.common.model.PictureHiBD;
import org.jcodec.scale.AWTUtil;
import ru.alexander.kernel.MovingKernel;
import ru.alexander.kernel.RenderKernel;
import ru.alexander.kernel.VelocityKernel;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Sandbox {
    private final double G;
    private final double minDeltaTime;
    private final double maxDeltaTime;

    private double deltaTime;


    private int len = 0;

    private double[] mass;
    private double[] radius;

    private int[] r;
    private int[] g;
    private int[] b;

    private double[] x;
    private double[] y;
    private double[] z;

    private double[] sx;
    private double[] sy;
    private double[] sz;

    private final VelocityKernel velocityKernel;
    private final MovingKernel movingKernel;

    private final RenderKernel renderKernel;
    private final Picture picture = Picture.create(1000, 1000, ColorSpace.RGB);
    public Sandbox(double gravityConstant, double minDeltaTime, double maxDeltaTime) throws CompileFailedException {
        G = gravityConstant;
        this.minDeltaTime = minDeltaTime;
        this.maxDeltaTime = maxDeltaTime;
        deltaTime = minDeltaTime;

        int size = 8;
        mass = new double[size];
        radius = new double[size];

        r = new int[size];
        g = new int[size];
        b = new int[size];

        x = new double[size];
        y = new double[size];
        z = new double[size];

        sx = new double[size];
        sy = new double[size];
        sz = new double[size];

        Device device = KernelManager.instance().bestDevice();

        velocityKernel = new VelocityKernel();
        velocityKernel.compile(device);

        movingKernel = new MovingKernel();
        movingKernel.compile(device);

        renderKernel = new RenderKernel(picture);
        renderKernel.compile(device);
    }

    public void addCelestial(Celestial celestial) {
        if (mass.length == len) {
            doubleBiplexing(mass);
            doubleBiplexing(radius);

            intBiplexing(r);
            intBiplexing(g);
            intBiplexing(b);

            doubleBiplexing(x);
            doubleBiplexing(y);
            doubleBiplexing(z);

            doubleBiplexing(sx);
            doubleBiplexing(sy);
            doubleBiplexing(sz);
        }
        mass[len] = celestial.getMass();
        radius[len] = celestial.getRadius();

        r[len] = celestial.getColor().getRed();
        g[len] = celestial.getColor().getGreen();
        b[len] = celestial.getColor().getBlue();

        x[len] = celestial.getPosition().x;
        y[len] = celestial.getPosition().y;
        z[len] = celestial.getPosition().z;

        sx[len] = celestial.getSpeed().x;
        sy[len] = celestial.getSpeed().y;
        sz[len] = celestial.getSpeed().z;
        len++;
    }

    public void render(double time, double shotTime, double nx, double ny, double nz, double zoom, File file) throws IOException {
        if (Math.abs(nx) < 1e-6) nx = 1e-6;
        if (Math.abs(ny) < 1e-6) ny = 1e-6;
        if (Math.abs(nz) < 1e-6) nz = 1e-6;
        if (Math.abs(zoom) < 1e-6) zoom = 1e-6;
        renderKernel.len = len;

        renderKernel.nx = nx;
        renderKernel.ny = ny;
        renderKernel.nz = nz;
        renderKernel.zoom = zoom;

        renderKernel.radius = radius;

        renderKernel.x = x;
        renderKernel.y = y;
        renderKernel.z = z;

        renderKernel.r = r;
        renderKernel.g = g;
        renderKernel.b = b;
        renderKernel.put(radius).put(r).put(g).put(b);
        SequenceEncoder video = SequenceEncoder.createSequenceEncoder(file, 60);
        double timer = 0;
        double st = 0;
        do {
            iteration();
            timer += deltaTime;
            st += deltaTime;
            if (st > shotTime) {
                renderKernel.put(x).put(y).put(z);
                renderKernel.execute(Range.create(picture.getWidth() * picture.getHeight()));
                renderKernel.get(renderKernel.bytes);
                video.encodeNativeFrame(picture);
                st = 0;
                System.out.println("Time: " + timer + "\tTime step: " + deltaTime);
            }
        } while (timer < time);
        video.finish();
        renderKernel.cleanUpArrays();
    }

    private void iteration() {
        //Velocity Calculating
        velocityKernel.deltaTime = deltaTime;
        velocityKernel.G = G;
        velocityKernel.len = len;

        velocityKernel.mass = mass;

        velocityKernel.x = x;
        velocityKernel.y = y;
        velocityKernel.z = z;

        velocityKernel.sx = sx;
        velocityKernel.sy = sy;
        velocityKernel.sz = sz;

        velocityKernel.error = new double[len];

        velocityKernel.execute(Range.create(len));

        sx = velocityKernel.sx;
        sy = velocityKernel.sy;
        sz = velocityKernel.sz;
        double error = 0;
        for (int i = 0; i < len; i++) error += velocityKernel.error[i];

        //Object Moving

        movingKernel.deltaTime = deltaTime;

        movingKernel.x = x;
        movingKernel.y = y;
        movingKernel.z = z;

        movingKernel.sx = sx;
        movingKernel.sy = sy;
        movingKernel.sz = sz;

        movingKernel.execute(Range.create(len));

        x = movingKernel.x;
        y = movingKernel.y;
        z = movingKernel.z;

        sx = movingKernel.sx;
        sy = movingKernel.sy;
        sz = movingKernel.sz;

        //Time Correction
        if (error != 0) {
            double dt = maxDeltaTime * Math.pow(maxDeltaTime * 1e-6 / (2 * error), 0.2);
            deltaTime = Math.max(minDeltaTime, Math.min(maxDeltaTime, dt));
        }
    }

    private void doubleBiplexing(double[] value) {
        double[] buff = new double[value.length * 2];
        System.arraycopy(value, 0, buff, 0, value.length);
        value = buff;
    }

    private void intBiplexing(int[] value) {
        int[] buff = new int[value.length * 2];
        System.arraycopy(value, 0, buff, 0, value.length);
        value = buff;
    }

    public double getDeltaTime() {
        return deltaTime;
    }

    public double firstCosmicalSpeed(int aroundIndex, double distance) {
        return Math.sqrt(G * mass[aroundIndex] / distance);
    }
}
