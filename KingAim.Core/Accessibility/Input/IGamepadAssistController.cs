namespace KingAim.Core.Accessibility.Input;

public interface IGamepadAssistController
{
    (float RightStickX, float RightStickY) Update(
        bool   hasTarget,
        float  errorX,
        float  errorY,
        float  targetVelocityX,
        float  targetVelocityY,
        float  confidence,
        double observationAgeMs,
        double dtSeconds,
        int?   trackId = null);

    void Reset();
}
