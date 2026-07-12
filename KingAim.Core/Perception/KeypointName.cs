namespace KingAim.Core.Perception;

/// <summary>
/// The four body landmarks used by all KingAim pose models.
/// Order is permanent — models are trained with this index mapping.
/// </summary>
public enum KeypointName
{
    Head       = 0,
    Neck       = 1,
    UpperChest = 2,
    Hip        = 3,
}
