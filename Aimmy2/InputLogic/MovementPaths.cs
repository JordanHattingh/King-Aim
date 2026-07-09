using System.Drawing;

namespace InputLogic
{
    class MovementPaths
    {
        private static readonly int[] permutation;

        static MovementPaths()
        {
            // Standard Perlin permutation: 0-255 shuffled with fixed seed, then doubled.
            // The previous all-zeros array produced constant-zero noise on every call.
            int[] p = new int[256];
            for (int i = 0; i < 256; i++) p[i] = i;
            var rng = new Random(42);
            for (int i = 255; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (p[i], p[j]) = (p[j], p[i]);
            }
            permutation = new int[512];
            for (int i = 0; i < 512; i++) permutation[i] = p[i & 255];
        }

        internal static Point CubicBezier(Point start, Point end, Point control1, Point control2, double t)
        {
            double u = 1 - t;
            double tt = t * t;
            double uu = u * u;

            double x = uu * u * start.X + 3 * uu * t * control1.X + 3 * u * tt * control2.X + tt * t * end.X;
            double y = uu * u * start.Y + 3 * uu * t * control1.Y + 3 * u * tt * control2.Y + tt * t * end.Y;

            return new Point((int)x, (int)y);
        }

        // Quadratic bezier with a perpendicular control-point offset so the cursor
        // follows a subtle arc rather than a straight line — matches real wrist motion.
        // offsetFactor: perpendicular displacement as a fraction of chord length (0.1–0.2 is natural).
        internal static Point HumanBezier(Point start, Point end, double t, double offsetFactor = 0.15)
        {
            double dx = end.X - start.X;
            double dy = end.Y - start.Y;
            double dist = Math.Sqrt(dx * dx + dy * dy);

            double cx, cy;
            if (dist < 4.0)
            {
                // Too small to arc — plain lerp avoids jitter on tiny corrections.
                cx = (start.X + end.X) * 0.5;
                cy = (start.Y + end.Y) * 0.5;
            }
            else
            {
                // Control point: midpoint shifted perpendicular (clockwise) by offsetFactor * dist.
                double perpX = -dy / dist;
                double perpY =  dx / dist;
                double offset = dist * offsetFactor;
                cx = (start.X + end.X) * 0.5 + perpX * offset;
                cy = (start.Y + end.Y) * 0.5 + perpY * offset;
            }

            // Quadratic bezier: P(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
            double u = 1.0 - t;
            double px = u * u * start.X + 2 * u * t * cx + t * t * end.X;
            double py = u * u * start.Y + 2 * u * t * cy + t * t * end.Y;
            return new Point((int)px, (int)py);
        }

        internal static Point Lerp(Point start, Point end, double t)
        {
            int x = (int)(start.X + (end.X - start.X) * t);
            int y = (int)(start.Y + (end.Y - start.Y) * t);
            return new Point(x, y);
        }

        internal static Point Exponential(Point start, Point end, double t, double exponent = 2.0)
        {
            double x = start.X + (end.X - start.X) * Math.Pow(t, exponent);
            double y = start.Y + (end.Y - start.Y) * Math.Pow(t, exponent);
            return new Point((int)x, (int)y);
        }

        // Short distances: lerp (no visible arc needed).
        // Long distances: HumanBezier arc for natural deceleration into target.
        internal static Point Adaptive(Point start, Point end, double t, double threshold = 80.0)
        {
            double distance = Math.Sqrt(Math.Pow(end.X - start.X, 2) + Math.Pow(end.Y - start.Y, 2));
            return distance < threshold
                ? Lerp(start, end, t)
                : HumanBezier(start, end, t, offsetFactor: 0.10);
        }

        internal static Point PerlinNoise(Point start, Point end, double t, double amplitude = 10.0, double frequency = 0.1)
        {
            double baseX = start.X + (end.X - start.X) * t;
            double baseY = start.Y + (end.Y - start.Y) * t;

            double noiseX = Noise(t * frequency, 0) * amplitude;
            double noiseY = Noise(t * frequency, 100) * amplitude;

            double perpX = -(end.Y - start.Y);
            double perpY = end.X - start.X;
            double perpLength = Math.Sqrt(perpX * perpX + perpY * perpY);

            if (perpLength > 0)
            {
                perpX /= perpLength;
                perpY /= perpLength;
            }

            double finalX = baseX + perpX * noiseX + noiseY * 0.3;
            double finalY = baseY + perpY * noiseX + noiseY * 0.3;

            return new Point((int)finalX, (int)finalY);
        }

        private static double Fade(double t)
        {
            return t * t * t * (t * (t * 6 - 15) + 10);
        }

        private static double Lerp(double a, double b, double t)
        {
            return a + t * (b - a);
        }

        private static double Grad(int hash, double x, double y)
        {
            int h = hash & 15;
            double u = h < 8 ? x : y;
            double v = h < 4 ? y : h == 12 || h == 14 ? x : 0;
            return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
        }

        private static double Noise(double x, double y)
        {
            int X = (int)Math.Floor(x) & 255;
            int Y = (int)Math.Floor(y) & 255;

            x -= Math.Floor(x);
            y -= Math.Floor(y);

            double u = Fade(x);
            double v = Fade(y);

            int A = permutation[X] + Y;
            int AA = permutation[A];
            int AB = permutation[A + 1];
            int B = permutation[X + 1] + Y;
            int BA = permutation[B];
            int BB = permutation[B + 1];

            return Lerp(Lerp(Grad(permutation[AA], x, y),
                           Grad(permutation[BA], x - 1, y), u),
                      Lerp(Grad(permutation[AB], x, y - 1),
                           Grad(permutation[BB], x - 1, y - 1), u), v);
        }
    }
}