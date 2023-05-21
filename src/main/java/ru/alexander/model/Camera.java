package ru.alexander.model;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import org.jcodec.api.SequenceEncoder;
import org.jcodec.common.model.ColorSpace;
import org.jcodec.common.model.Picture;

import java.io.File;
import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;

public class Camera {
    //      cd C:\Projects\JavaProjects\FastStarSimulation\src\main\resources & nvcc -ptx -m64 -arch=native render.cu -o render.ptx
    //      cd C:\Projects\JavaProjects\FastStarSimulation\src\main\resources & nvcc -fatbin -m64 -arch=all-major render.cu -o render.fatbin
    private final Picture picture;
    private final Sandbox sandbox;

    private final CUdeviceptr picturePointer;


    private final CUmodule module;
    private final CUfunction render;
    private final long size;


    public double cameraX = 0;
    public double cameraY = 0;
    public double cameraZ = 0;

    public double scale = 0;

    public Camera(Sandbox sandbox, int width, int height) {
        this.sandbox = sandbox;
        picture = Picture.create(width, height, ColorSpace.RGB);

        size = (long) width * height * 3 * Sizeof.BYTE;
        picturePointer = new CUdeviceptr();
        cuMemAlloc(picturePointer, size);

        module = new CUmodule();
        cuModuleLoad(module, "src/main/resources/render.ptx");

        render = new CUfunction();
        cuModuleGetFunction(render, module, "render");
    }

    private void shot() {
        int blockSize = 32;

        sandbox.getX().load();
        sandbox.getY().load();
        sandbox.getZ().load();

        sandbox.getR().load();
        sandbox.getG().load();
        sandbox.getB().load();

        cuLaunchKernel(render,
                (int) Math.ceil((double) picture.getWidth() / blockSize),  (int) Math.ceil((double) picture.getHeight() / blockSize), 1,
                Math.min(picture.getWidth(), blockSize), Math.min(picture.getHeight(), blockSize), 1,
                0, null,
                Pointer.to(
                        Pointer.to(new int[] { picture.getWidth() }),
                        Pointer.to(new int[] { picture.getHeight() }),
                        Pointer.to(new int[] { sandbox.getX().size() }),
                        Pointer.to(new double[] { scale }),

                        Pointer.to(new double[] { cameraX }),
                        Pointer.to(new double[] { cameraY }),
                        Pointer.to(new double[] { cameraZ }),

                        Pointer.to(sandbox.getMass().getPointer()),
                        Pointer.to(sandbox.getDensity().getPointer()),

                        Pointer.to(sandbox.getX().getPointer()),
                        Pointer.to(sandbox.getY().getPointer()),
                        Pointer.to(sandbox.getZ().getPointer()),

                        Pointer.to(sandbox.getR().getPointer()),
                        Pointer.to(sandbox.getG().getPointer()),
                        Pointer.to(sandbox.getB().getPointer()),

                        Pointer.to(picturePointer)
                ),
                null);
        cuCtxSynchronize();


        sandbox.getX().unload();
        sandbox.getY().unload();
        sandbox.getZ().unload();

        sandbox.getR().unload();
        sandbox.getG().unload();
        sandbox.getB().unload();

        cuMemcpyDtoH(Pointer.to(picture.getPlaneData(0)), picturePointer, size);
    }

    public void render(double from, double to, double shotStep, double simulationStep, File output, int fps) throws IOException {
        double step = 0;
        System.out.print("Precalculation...");
        while (step < from) {
            sandbox.calculate(simulationStep);
            step += simulationStep;
        }
        System.out.println("\rPrecalculation completed!");
        System.out.println("Render...");

        SequenceEncoder encoder = SequenceEncoder.createSequenceEncoder(output, fps);
        double shotTime = 0;
        while (step < to) {
            sandbox.calculate(simulationStep);
            step += simulationStep;

            if (shotTime > shotStep) {
                shot();
                encoder.encodeNativeFrame(picture);
                shotTime = 0;
            }
            shotTime += simulationStep;
        }
        encoder.finish();
        System.out.println("Render completed!");
    }
}
