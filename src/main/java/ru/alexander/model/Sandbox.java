package ru.alexander.model;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuCtxCreate;

public class Sandbox {
    private final PointedArrayList mass;
    private final PointedArrayList density;

    private final PointedArrayList vx;
    private final PointedArrayList vy;
    private final PointedArrayList vz;

    private final PointedArrayList x;
    private final PointedArrayList y;
    private final PointedArrayList z;



    private final CUcontext context;
    private final CUdevice device;
    private final CUmodule module;


    private final CUfunction accelerate;
    private final CUfunction move;

    private final CUfunction collide;
    private final CUfunction merge;


    public Sandbox() {
        this(8);
    }

    public Sandbox(int startCelestialCount) {
        mass = new PointedArrayList(startCelestialCount);
        density = new PointedArrayList(startCelestialCount);

        vx = new PointedArrayList(startCelestialCount);
        vy = new PointedArrayList(startCelestialCount);
        vz = new PointedArrayList(startCelestialCount);

        x = new PointedArrayList(startCelestialCount);
        y = new PointedArrayList(startCelestialCount);
        z = new PointedArrayList(startCelestialCount);

        setExceptionsEnabled(true);

        cuInit(0);
        device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
        module = new CUmodule();
        cuModuleLoad(module, "src/main/resources/gravity.cubin");

        accelerate = new CUfunction();
        cuModuleGetFunction(accelerate, module, "accelerate");

        move = new CUfunction();
        cuModuleGetFunction(move, module, "move");

        collide = new CUfunction();
        cuModuleGetFunction(collide, module, "collide");

        merge = new CUfunction();
        cuModuleGetFunction(merge, module, "merge");
    }

    public void destroy() {
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }


    public void addCelestial(double mass, double density,
                             double x, double y, double z,
                             double vx, double vy, double vz) {
        this.mass.add(mass);
        this.density.add(density);

        this.vx.add(vx);
        this.vy.add(vy);
        this.vz.add(vz);

        this.x.add(x);
        this.y.add(y);
        this.z.add(z);
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
    }

    public void calculate(double dt) {
        int blockSize = 1024;
        cuLaunchKernel(accelerate,
                (int) Math.ceil((double) mass.size() / blockSize), 1, 1,
                Math.min(mass.size(), blockSize), 1, 1,
                0, null,
                Pointer.to(
                        Pointer.to(new int[] { mass.size() }),
                        mass.getPointer(),

                        x.getPointer(),
                        y.getPointer(),
                        z.getPointer(),

                        vx.getPointer(),
                        vy.getPointer(),
                        vz.getPointer()
                ),
                null);

        cuLaunchKernel(move,
                (int) Math.ceil((double) mass.size() / blockSize), 1, 1,
                Math.min(mass.size(), blockSize), 1, 1,
                0, null,
                Pointer.to(
                        Pointer.to(new int[] { mass.size() }),

                        x.getPointer(),
                        y.getPointer(),
                        z.getPointer(),

                        vx.getPointer(),
                        vy.getPointer(),
                        vz.getPointer()
                ),
                null);

        int collisionMatrixSize = mass.size() * (mass.size() - 1) / 2;
        cuLaunchKernel(collide,
                (int) Math.ceil((double) collisionMatrixSize / blockSize), 1, 1,
                Math.min(collisionMatrixSize, blockSize), 1, 1,
                0, null,
                Pointer.to(
                        Pointer.to(new int[] { mass.size() }),
                        mass.getPointer(),
                        density.getPointer(),

                        x.getPointer(),
                        y.getPointer(),
                        z.getPointer()
                ),
                null);

    }
}
