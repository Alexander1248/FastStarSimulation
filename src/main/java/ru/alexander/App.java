package ru.alexander;

import com.aparapi.exception.CompileFailedException;
import ru.alexander.model.PointedArrayList;

import java.awt.*;
import java.io.IOException;

public class App {
    public static void main(String[] args) throws CompileFailedException, IOException {
    }

//    private static void galaxy(Sandbox sandbox, Vector3 position) {
//        sandbox.addCelestial(new Celestial(1e5, 10, new Color(32, 32, 32), position));
//        for (int i = 0; i < 1023; i++) {
//            double dst = 100 * Math.sqrt(0.03 + Math.random() * 0.97);
//            double h = (Math.random() * 2 - 1) * (1 / (1 + Math.pow(dst / 50, 2)) - 0.2) * 12.5;
//            double sdst = Math.sqrt(dst * dst + h * h);
//
//            double speed = sandbox.firstCosmicalSpeed(0, sdst) * Math.sqrt(1 + Math.random() / 10);
//            double rx = Math.random() * Math.PI * 2;
//
//            Vector3 pos = new Vector3(position.x + dst * Math.cos(rx), position.y + dst * Math.sin(rx), position.z + h);
//            Vector3 spd = new Vector3(-speed * Math.sin(rx), speed * Math.cos(rx), 0);
//
//            sandbox.addCelestial(new Celestial(1, 1, thermo(700 + Math.random() * 30000), pos, spd));
//        }
//    }

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
