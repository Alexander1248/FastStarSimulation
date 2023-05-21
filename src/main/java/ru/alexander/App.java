package ru.alexander;

import com.aparapi.exception.CompileFailedException;
import ru.alexander.model.Sandbox;

import java.awt.*;
import java.io.IOException;

public class App {


    public static void main(String[] args) {
        Sandbox sandbox = new Sandbox(8, 100);
        galaxy(sandbox, 0, 0, 0);

        int count = 0;
        while (true) {
            sandbox.calculate(1e-4);
            if (count > 1000) {
                System.out.println(sandbox.getX().size());
                count = 0;
            }
            count++;
        }
    }

    private static void galaxy(Sandbox sandbox, double x, double y, double z) {
        sandbox.addCelestial(1e5, 1e4, x, y, z, 0, 0, 0, 0.125, 0.125, 0.125);
        for (int i = 0; i < 1023; i++) {
            double dst = 100 * Math.sqrt(0.03 + Math.random() * 0.97);
            double h = (Math.random() * 2 - 1) * (1 / (1 + Math.pow(dst / 50, 2)) - 0.2) * 12.5;
            double sdst = Math.sqrt(dst * dst + h * h);

            double speed = sandbox.firstCosmicalSpeed(0, sdst) * Math.sqrt(1 + Math.random() / 10);
            double rx = Math.random() * Math.PI * 2;

            double cx = x + dst * Math.cos(rx);
            double cy = y + dst * Math.sin(rx);
            double cz = z + h;
            double vx = -speed * Math.sin(rx);
            double vy = speed * Math.cos(rx);

            Color color = thermo(1000 + Math.random() * 12000);
            double r = (double) color.getRed() / 255;
            double g = (double) color.getGreen() / 255;
            double b = (double) color.getBlue() / 255;
            sandbox.addCelestial(4.18935, 1, cx, cy, cz, vx, vy, 0, r, g, b);
        }
    }

    public static Color thermo(double temperature) {
        temperature /= 100;
        double r;
        double g;
        double b;

        if (temperature <= 66) {
            g =  Math.max(0, Math.min(255, 99.4708025861 * Math.log(temperature) - 161.1195681661));
            if (temperature <= 19) {
                r = Math.max(0, Math.min(255, Math.PI * temperature * temperature));
                b = 0;
            }
            else {
                r = 255;
                double temp = temperature - 10;
                b = Math.max(0, Math.min(255, 138.5177312231 * Math.log(temp) - 305.0447927307));
            }
        }
        else {
            double temp = temperature - 60;
            r = Math.max(0, Math.min(255, 329.698727446 * Math.pow(temp, -0.1332047592)));
            g = Math.max(0, Math.min(255, 288.1221695283 * Math.pow(temp, -0.0755148492)));
            b = 255;
        }


        return new Color((int) Math.round(r), (int) Math.round(g), (int) Math.round(b));
    }
}
