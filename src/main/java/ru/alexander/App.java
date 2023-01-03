package ru.alexander;

import com.aparapi.exception.CompileFailedException;
import ru.alexander.objects.Celestial;
import ru.alexander.objects.Sandbox;
import ru.alexander.objects.Vector3;

import java.awt.*;
import java.io.File;
import java.io.IOException;

public class App {
    public static void main(String[] args) throws CompileFailedException, IOException {
        Sandbox sandbox = new Sandbox(1, 1e-2, 0);
        sandbox.addCelestial(new Celestial(100, 10, Color.yellow));
        sandbox.addCelestial(new Celestial(1, 5, Color.gray,
                new Vector3(0,100, 0),
                new Vector3(sandbox.firstCosmicalSpeed(0, 100), 0, 0)));
        sandbox.render(500, 1,0,0,1, 5, new File("test.mp4"));
    }
}
