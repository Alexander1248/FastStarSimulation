package ru.alexander.model;

import jcuda.Pointer;
import jcuda.driver.*;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuCtxCreate;

public class Sandbox {
    //      cd C:\Projects\JavaProjects\FastStarSimulation\src\main\resources & nvcc -ptx -m64 -arch=native gravity.cu -o gravity.ptx
    //      cd C:\Projects\JavaProjects\FastStarSimulation\src\main\resources & nvcc -fatbin -m64 -arch=all-major gravity.cu -o gravity.fatbin
    private final PointedArrayList mass;
    private final PointedArrayList density;

    private final PointedArrayList vx;
    private final PointedArrayList vy;
    private final PointedArrayList vz;

    private final PointedArrayList x;
    private final PointedArrayList y;
    private final PointedArrayList z;

    private final PointedArrayList r;
    private final PointedArrayList g;
    private final PointedArrayList b;



    private final CUcontext context;
    private final CUmodule module;


    private final CUfunction accelerate;
    private final CUfunction move;

    private final CUfunction collide;

    private final double G;


    public Sandbox() {
        this(8, 6.67e-11);
    }
    public Sandbox(double G) {
        this(8, G);
    }

    public Sandbox(int startCelestialCount, double G) {
        this.G = G;

        mass = new PointedArrayList(startCelestialCount);
        density = new PointedArrayList(startCelestialCount);

        vx = new PointedArrayList(startCelestialCount);
        vy = new PointedArrayList(startCelestialCount);
        vz = new PointedArrayList(startCelestialCount);

        x = new PointedArrayList(startCelestialCount);
        y = new PointedArrayList(startCelestialCount);
        z = new PointedArrayList(startCelestialCount);

        r = new PointedArrayList(startCelestialCount);
        g = new PointedArrayList(startCelestialCount);
        b = new PointedArrayList(startCelestialCount);

        setExceptionsEnabled(true);

        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
        module = new CUmodule();
        cuModuleLoad(module, "src/main/resources/gravity.ptx");

        accelerate = new CUfunction();
        cuModuleGetFunction(accelerate, module, "accelerate");

        move = new CUfunction();
        cuModuleGetFunction(move, module, "move");

        collide = new CUfunction();
        cuModuleGetFunction(collide, module, "collide");
    }

    public void destroy() {
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }


    public void addCelestial(double mass, double density,
                             double x, double y, double z,
                             double vx, double vy, double vz,
                             double r, double g, double b) {
        this.mass.add(mass);
        this.density.add(density);

        this.vx.add(vx);
        this.vy.add(vy);
        this.vz.add(vz);

        this.x.add(x);
        this.y.add(y);
        this.z.add(z);

        this.r.add(r);
        this.g.add(g);
        this.b.add(b);
    }

    public void removeCelestial(int index) {
        mass.remove(index);
        density.remove(index);

        vx.remove(index);
        vy.remove(index);
        vz.remove(index);

        x.remove(index);
        y.remove(index);
        z.remove(index);

        r.remove(index);
        g.remove(index);
        b.remove(index);
    }

    public void calculate(double dt) {
        int blockSize = 16;

        mass.load();
        density.load();

        x.load();
        y.load();
        z.load();

        vx.load();
        vy.load();
        vz.load();

        cuLaunchKernel(accelerate,
                (int) Math.ceil((double) mass.size() / blockSize), 1, 1,
                Math.min(mass.size(), blockSize), 1, 1,
                0, null,
                Pointer.to(
                        Pointer.to(new int[] { mass.size() }),
                        Pointer.to(new double[] { dt }),
                        Pointer.to(new double[] { G }),
                        Pointer.to(mass.getPointer()),

                        Pointer.to(x.getPointer()),
                        Pointer.to(y.getPointer()),
                        Pointer.to(z.getPointer()),

                        Pointer.to(vx.getPointer()),
                        Pointer.to(vy.getPointer()),
                        Pointer.to(vz.getPointer())
                ),
                null);
        cuCtxSynchronize();

        cuLaunchKernel(move,
                (int) Math.ceil((double) mass.size() / blockSize), 1, 1,
                Math.min(mass.size(), blockSize), 1, 1,
                0, null,
                Pointer.to(
                        Pointer.to(new int[] { mass.size() }),
                        Pointer.to(new double[] { dt }),

                        Pointer.to(x.getPointer()),
                        Pointer.to(y.getPointer()),
                        Pointer.to(z.getPointer()),

                        Pointer.to(vx.getPointer()),
                        Pointer.to(vy.getPointer()),
                        Pointer.to(vz.getPointer())
                ),
                null);
        cuCtxSynchronize();

        int collisionMatrixSize = mass.size() * (mass.size() - 1) / 2;
        cuLaunchKernel(collide,
                (int) Math.ceil((double) collisionMatrixSize / blockSize), 1, 1,
                Math.min(collisionMatrixSize, blockSize), 1, 1,
                0, null,
                Pointer.to(
                        Pointer.to(new int[] { mass.size() }),
                        Pointer.to(mass.getPointer()),
                        Pointer.to(density.getPointer()),

                        Pointer.to(x.getPointer()),
                        Pointer.to(y.getPointer()),
                        Pointer.to(z.getPointer())
                ),
                null);
        cuCtxSynchronize();

        mass.unload();
        density.unload();

        x.unload();
        y.unload();
        z.unload();

        vx.unload();
        vy.unload();
        vz.unload();

        for (int i = 0; i < mass.size(); i++) {
            if (mass.get(i) == 0) {
               removeCelestial(i);
                i--;
            }
        }

    }

    public double firstCosmicalSpeed(int i, double dst) {
        return Math.sqrt(G * mass.get(i) / dst);
    }

    public PointedArrayList getMass() {
        return mass;
    }

    public PointedArrayList getDensity() {
        return density;
    }

    public PointedArrayList getVx() {
        return vx;
    }

    public PointedArrayList getVy() {
        return vy;
    }

    public PointedArrayList getVz() {
        return vz;
    }

    public PointedArrayList getX() {
        return x;
    }

    public PointedArrayList getY() {
        return y;
    }

    public PointedArrayList getZ() {
        return z;
    }

    public PointedArrayList getR() {
        return r;
    }

    public PointedArrayList getG() {
        return g;
    }

    public PointedArrayList getB() {
        return b;
    }
}
