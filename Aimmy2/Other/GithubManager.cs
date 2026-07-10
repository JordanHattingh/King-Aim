using Newtonsoft.Json;
using System.Linq;
using System.Net.Http;
using System.Text.Json;

namespace Aimmy2.Other
{
    public class GithubManager
    {
        private readonly HttpClient httpClient;

        public GithubManager()
        {
            httpClient = new HttpClient();
            httpClient.DefaultRequestHeaders.Add("User-Agent", "Aimmy2");
            httpClient.DefaultRequestHeaders.Add("Accept", "application/vnd.github.v3+json");
        }

        private class GitHubContent
        {
            public string? name { get; set; }
        }

        public async Task<(string tagName, string downloadUrl)> GetLatestReleaseInfo(string owner, string repo)
        {
            string apiUrl = $"https://api.github.com/repos/{owner}/{repo}/releases/latest";

            using var response = await httpClient.GetAsync(apiUrl);
            response.EnsureSuccessStatusCode();

            var content = await response.Content.ReadAsStringAsync();
            var data = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(content)
                ?? throw new InvalidOperationException("GitHub release API returned a null or empty response.");

            if (!data.TryGetValue("tag_name", out var tagNameObj) || tagNameObj is not JsonElement tagNameEl)
                throw new InvalidOperationException("GitHub release API response is missing 'tag_name'.");
            string tagName = tagNameEl.GetString()
                ?? throw new InvalidOperationException("GitHub release API 'tag_name' value is null.");

            if (!data.TryGetValue("assets", out var assetsObj) || assetsObj is not JsonElement assetsEl)
                throw new InvalidOperationException("GitHub release API response is missing 'assets'.");

            var assets = assetsEl.EnumerateArray().ToList();
            if (assets.Count == 0)
                throw new InvalidOperationException("GitHub release API response contains no assets.");

            if (!assets[0].TryGetProperty("browser_download_url", out var urlEl))
                throw new InvalidOperationException("GitHub release asset is missing 'browser_download_url'.");
            string downloadUrl = urlEl.GetString()
                ?? throw new InvalidOperationException("GitHub release asset 'browser_download_url' is null.");

            return (tagName, downloadUrl);
        }

        public async Task<IEnumerable<string?>> FetchGithubFilesAsync(string url)
        {
            var response = await httpClient.GetAsync(url);

            response.EnsureSuccessStatusCode();

            var content = await response.Content.ReadAsStringAsync();

            List<GitHubContent>? contents = JsonConvert.DeserializeObject<List<GitHubContent>>(content);
            if (contents == null)
            {
                throw new InvalidOperationException("Failed to deserialize GitHub content or Github content is empty.");
            }

            return contents.Select(c => c.name);
        }

        public void Dispose()
        {
            httpClient.Dispose();
        }
    }
}