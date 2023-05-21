package ru.alexander.model;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.*;

public class PointedArrayList {
    private double[] data;
    private int count = 0;

    private int pointerSize = 0;
    private final CUdeviceptr pointer = new CUdeviceptr();

    public PointedArrayList() {
        this(8);
    }

    public PointedArrayList(int size) {
        data = new double[Math.max(1, size)];
    }

    public void add(double value) {
        if (count == data.length) {
            double[] buffer = new double[data.length * 2];
            System.arraycopy(data, 0, buffer, 0, data.length);
            data = buffer;
        }
        data[count] = value;
        count++;
    }
    public void remove(int index) {
        if (index >= 0 && index < count) {
            count--;
            for (int i = index; i < count; i++)
                data[i] = data[i + 1];
        }
        else throw new IndexOutOfBoundsException("Size: " + count + " Index: " + index);
    }

    public double get(int index) {
        if (index >= 0 && index < count) return data[index];
        else throw new IndexOutOfBoundsException("Size: " + count + " Index: " + index);
    }
    public void set(int index, double value) {
        if (index >= 0 && index < count) data[index] = value;
        else throw new IndexOutOfBoundsException("Size: " + count + " Index: " + index);
    }



    public void load() {
        recalculatePointer();
        cuMemcpyHtoD(pointer, Pointer.to(data), (long) pointerSize * Sizeof.DOUBLE);
    }

    public void unload() {
        recalculatePointer();
        cuMemcpyDtoH(Pointer.to(data), pointer, (long) pointerSize * Sizeof.DOUBLE);

    }

    public CUdeviceptr getPointer() {
        recalculatePointer();
        return pointer;
    }

    private void recalculatePointer() {
        if (pointerSize != data.length) {
            pointerSize = data.length;
            cuMemFree(pointer);
            cuMemAlloc(pointer, (long) pointerSize * Sizeof.DOUBLE);
        }
    }

    public double[] getArray() {
        return data;
    }

    @Override
    public String toString() {
        return Arrays.toString(Arrays.copyOf(data, count));
    }

    public int size() {
        return count;
    }
}
