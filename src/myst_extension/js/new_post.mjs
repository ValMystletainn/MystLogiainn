import https from 'https';
// Replace these with the repository information
const owner = 'ValMystletainn';  // TODO set it by import args
const repo = 'ob-notes';
const branch = 'master';
// GitHub API endpoint for listing commits in the master branch
const commitsUrl = `/repos/${owner}/${repo}/commits?sha=${branch}&per_page=100`;
// Function to make HTTPS GET requests
function httpsGet(url) {
const options = {
    hostname: 'api.github.com',
    path: url,
    method: 'GET',
    headers: {
    'User-Agent': 'mystblog',
    'Accept': 'application/vnd.github.v3+json'
    // 'Authorization': `token YOUR_GITHUB_ACCESS_TOKEN` // Uncomment if using a private repo or higher rate limits
    }
};
return new Promise((resolve, reject) => {
    const req = https.request(options, res => {
    let data = '';
    res.on('data', chunk => {
        data += chunk;
    });
    res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
        resolve({ data: JSON.parse(data), headers: res.headers });
        } else {
        reject(new Error(`Request failed with status code ${res.statusCode}: ${data}`));
        }
    });
    });
    req.on('error', reject);
    req.end();
});
}
// Function to get commits from the master branch
async function getCommits(url) {
let allCommits = [];
while (url) {
    const { data, headers } = await httpsGet(url);
    allCommits = allCommits.concat(data);
    // Check for pagination
    const linkHeader = headers.link;
    if (linkHeader) {
    const links = linkHeader.split(',').map(link => link.trim());
    const nextLink = links.find(link => link.includes('rel="next"'));
    if (nextLink) {
        url = nextLink.split(';')[0].slice(1, -1);
    } else {
        url = null;
    }
    } else {
    url = null;
    }
}
return allCommits;
}
// Function to get new files from a specific commit and its timestamp
async function getNewFilesFromCommit(commitSha) {
const commitUrl = `/repos/${owner}/${repo}/commits/${commitSha}`;
const { data } = await httpsGet(commitUrl);
const commitData = data.commit;
const commitTimestamp = commitData.committer.date;
const files = data.files || [];
const newFiles = files.filter(file => file.status === 'added').map(file => ({
    filename: file.filename,
    commitTimestamp
}));
return newFiles;
}
async function main() {
try {
    const allNewFiles = [];
    const commits = await getCommits(commitsUrl);
    for (const commit of commits) {
        const commitSha = commit.sha;
        const newFiles = await getNewFilesFromCommit(commitSha);
        allNewFiles.push(...newFiles);
    }
    if (allNewFiles.length > 0) {
    console.log('All new files added in the master branch with their timestamps:');
    allNewFiles.forEach(file => {
        console.log(`File: ${file.filename}, Timestamp: ${file.commitTimestamp}`);
    });
    } else {
    console.log('No new files found in the master branch.');
    }
} catch (error) {
    console.error('Error:', error.message);
}
}


const NewPostDirective = {
    name: 'new-post',
    doc: 'get the new file list from the github repo',
    async run(data) {
        const allNewFiles = [];
        const commits = await getCommits(commitsUrl);
        for (const commit of commits) {
            const commitSha = commit.sha;
            const newFiles = await getNewFilesFromCommit(commitSha);
            allNewFiles.push(...newFiles);
        }

        const test_string = allNewFiles[0];
        const postText = test_string.toString();
        console.log(postText)

        const paragraph = { 
            type: 'text',
            value: postText,
        };
        return [paragraph];
    },
};

const plugin = { name: 'new post', directives: [NewPostDirective] };

export default plugin;
