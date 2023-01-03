package ru.alexander.objects;

import java.awt.*;

public class Celestial {
    private final double mass;
    private final double radius;

    private final Color color;

    private final Vector3 position;
    private final Vector3 speed;

    public Celestial(double mass, double radius, Color color, Vector3 position, Vector3 speed) {
        this.mass = mass;
        this.radius = radius;
        this.color = color;
        this.position = position;
        this.speed = speed;
    }

    public Celestial(double mass, double radius, Color color, Vector3 position) {
        this(mass, radius, color, position, new Vector3(0, 0, 0));
    }
    public Celestial(double mass, double radius, Color color) {
        this(mass, radius, color, new Vector3(0, 0, 0), new Vector3(0, 0, 0));
    }

    public double getMass() {
        return mass;
    }

    public double getRadius() {
        return radius;
    }

    public Color getColor() {
        return color;
    }

    public Vector3 getPosition() {
        return position;
    }

    public Vector3 getSpeed() {
        return speed;
    }
}
