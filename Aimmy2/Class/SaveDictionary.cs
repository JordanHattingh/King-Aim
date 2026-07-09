using Newtonsoft.Json;
using Other;
using System.IO;
using MessageBox = System.Windows.MessageBox;

namespace Class
{
    internal class SaveDictionary
    {
        // All user-writable data lives under %AppData%\King Aim so the app works correctly
        // when installed to C:\Program Files\ (write-protected by Windows UAC).
        private static readonly string AppDataDir =
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "King Aim");

        /// <summary>
        /// Resolves a relative path like "bin\colors.cfg" to the writable AppData folder.
        /// Absolute paths are returned unchanged so callers that already use full paths still work.
        /// </summary>
        public static string ResolvePath(string path)
        {
            if (Path.IsPathRooted(path))
                return path;
            return Path.Combine(AppDataDir, path);
        }

        // Ensure all required directories exist at startup
        public static void EnsureDirectoriesExist()
        {
            var requiredDirectories = new[]
            {
                "bin",
                "bin\\configs",
                "bin\\labels",
                "bin\\models"
            };

            foreach (var dir in requiredDirectories)
            {
                string fullDir = ResolvePath(dir);
                if (!Directory.Exists(fullDir))
                {
                    try
                    {
                        Directory.CreateDirectory(fullDir);
                    }
                    catch (Exception ex)
                    {
                        LogManager.Log(LogManager.LogLevel.Error, $"Failed to create directory {fullDir}: {ex.Message}", true);
                    }
                }
            }
        }

        public static void WriteJSON(Dictionary<string, dynamic> dictionary, string path = "bin\\configs\\Default.cfg", string SuggestedModel = "", string ExtraStrings = "")
        {
            path = ResolvePath(path);
            try
            {
                // Ensure the directory exists
                string? directory = Path.GetDirectoryName(path);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                var SavedJSONSettings = new Dictionary<string, dynamic>(dictionary);
                if (!string.IsNullOrEmpty(SuggestedModel) && SavedJSONSettings.ContainsKey("Suggested Model"))
                {
                    SavedJSONSettings["Suggested Model"] = SuggestedModel + ".onnx" + ExtraStrings;
                }

                string json = JsonConvert.SerializeObject(SavedJSONSettings, Formatting.Indented);
                File.WriteAllText(path, json);
            }
            catch (Exception ex)
            {
                // Only show error if it's not a directory creation issue
                MessageBox.Show($"Error writing JSON, please note:\n{ex}");
            }
        }

        public static void LoadJSON(Dictionary<string, dynamic> dictionary, string path = "bin\\configs\\Default.cfg", bool strict = true)
        {
            path = ResolvePath(path);
            try
            {
                // Ensure the directory exists before checking for the file
                string? directory = Path.GetDirectoryName(path);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                if (!File.Exists(path))
                {
                    WriteJSON(dictionary, path);
                    return;
                }

                var configuration = JsonConvert.DeserializeObject<Dictionary<string, dynamic>>(File.ReadAllText(path));
                if (configuration == null) return;

                foreach (var (key, value) in configuration)
                {
                    if (dictionary.ContainsKey(key))
                    {
                        dictionary[key] = value;
                    }
                    else if (!strict)
                    {
                        dictionary.Add(key, value);
                    }
                }
            }
            catch (Exception ex)
            {
                // If there's an error loading, try to recreate the file with defaults
                try
                {
                    WriteJSON(dictionary, path);
                }
                catch
                {
                    // Only show error if we can't even create a default file
                    MessageBox.Show("Error loading JSON, please note:\n" + ex.ToString());
                }
            }
        }
    }
}