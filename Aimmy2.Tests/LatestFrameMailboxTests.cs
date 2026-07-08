using System.Drawing;
using Aimmy2.AILogic;
using Xunit;

namespace Aimmy2.Tests
{
    public class LatestFrameMailboxTests
    {
        private static CapturedFrame MakeFrame(long id) => new(
            FrameId: id,
            CaptureStartedAt: DateTime.UtcNow,
            CaptureCompletedAt: DateTime.UtcNow,
            Width: 4,
            Height: 4,
            CaptureRegion: new Rectangle(0, 0, 4, 4),
            Image: new Bitmap(4, 4));

        [Fact]
        public void LatestFrameMailbox_ReplacesOldFrame()
        {
            using var mailbox = new LatestFrameMailbox();

            mailbox.Post(MakeFrame(1));
            mailbox.Post(MakeFrame(2));
            mailbox.Post(MakeFrame(3));

            bool took = mailbox.TryTake(out var frame);

            Assert.True(took);
            Assert.NotNull(frame);
            Assert.Equal(3, frame!.FrameId);

            bool tookAgain = mailbox.TryTake(out var second);
            Assert.False(tookAgain);
            Assert.Null(second);

            Assert.Equal(3, mailbox.FramesCaptured);
            Assert.Equal(2, mailbox.FramesReplaced);
            Assert.Equal(1, mailbox.FramesConsumed);

            frame.Image.Dispose();
        }

        [Fact]
        public void LatestFrameMailbox_DisposesReplacedFrames()
        {
            using var mailbox = new LatestFrameMailbox();

            var first = MakeFrame(1);
            var second = MakeFrame(2);

            mailbox.Post(first);
            mailbox.Post(second);

            // The first frame's bitmap should have been disposed when replaced;
            // accessing a disposed Bitmap's properties throws ArgumentException.
            Assert.Throws<ArgumentException>(() => _ = first.Image.Width);

            mailbox.TryTake(out var taken);
            taken!.Image.Dispose();
        }

        [Fact]
        public void Dispose_DisposesHeldFrame()
        {
            var frame = MakeFrame(1);
            var mailbox = new LatestFrameMailbox();
            mailbox.Post(frame);

            mailbox.Dispose();

            Assert.Throws<ArgumentException>(() => _ = frame.Image.Width);
        }
    }
}
