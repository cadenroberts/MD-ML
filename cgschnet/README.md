# CGSchNet User Guide

## Setting Up Git for md-ml Users

### Accessing the Server as md-ml User

Connect to the server with the following command:
```bash
ssh -X md-ml@raz.ucsc.edu
```

### Generating an SSH Key

Generate a unique SSH key for your account. Replace `YOUR_USERNAME` and `YOUR_EMAIL` with your details:
```bash
ssh-keygen -t rsa -b 4096 -C "YOUR_EMAIL" -f ~/.ssh/id_rsa.YOUR_USERNAME
```
This creates a private key named `id_rsa.YOUR_USERNAME` in your `~/.ssh/` directory.

### Registering with GitHub

``Add your SSH key to GitHub. `` 

Replace `YOUR_USERNAME` with your username:
```bash
cat ~/.ssh/id_rsa.YOUR_USERNAME.pub
```
Copy the displayed public key and add it to your GitHub SSH keys.

### Configuring SSH

Edit the `~/.ssh/config` file for GitHub configuration. Replace `YOUR_GITHUB_ALIAS` and `YOUR_USERNAME` with your specific details:
```bash
Host YOUR_GITHUB_ALIAS
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_rsa.YOUR_USERNAME
  IdentitiesOnly yes
```

### Verifying Connection

Verify your SSH connection to GitHub using your alias:
```bash
ssh -T YOUR_GITHUB_ALIAS
```
You should see a successful authentication message.

### Setting Up `.gitconfig`

Create and configure your work directory. Replace `YOUR_DIRECTORY` and `YOUR_USERNAME` appropriately:
```bash
mkdir ~/YOUR_DIRECTORY
touch ~/.gitconfig_YOUR_USERNAME
```
Add your user details to `.gitconfig_YOUR_USERNAME`. Replace `YOUR_GITHUB_USERNAME` and `YOUR_EMAIL`:
```bash
[user]
    name = YOUR_GITHUB_USERNAME
    email = YOUR_EMAIL
```
Link this configuration in `~/.gitconfig`:
```bash
[includeIf "gitdir:~/YOUR_DIRECTORY/"]
    path = ~/.gitconfig_YOUR_USERNAME
```

### Cloning the Repository

Clone the repository into your directory. Replace `YOUR_DIRECTORY` and `YOUR_GITHUB_ALIAS`:
```bash
cd ~/YOUR_DIRECTORY
git clone git@YOUR_GITHUB_ALIAS:BioMedAI-UCSC/cgschnet.git
```

After cloning, move to the repository directory:
```bash
cd cgschnet
```
### Checking User Configuration

Inside the repository directory, check your GitHub configuration:
```bash
git config user.name
```

## Working with Git

### Branch Management

Create and switch to your branch. Replace `YOUR_BRANCH_NAME` with your desired branch name:
```bash
git branch YOUR_BRANCH_NAME
git checkout YOUR_BRANCH_NAME
```
> NOTE: Your branch will not appear on Github until you push (see below section)

### Developing, Staging, and Pushing Changes

Make your changes, then stage and commit them. Replace `YOUR_COMMIT_MESSAGE` with your commit message:
```bash
git add .
git commit -m "YOUR_COMMIT_MESSAGE"
```

Once you're ready to push your code to Github type in:
```bash
git push origin YOUR_BRANCH_NAME
```

### Monitoring Branch Status

Regularly check your current branch:
```bash
git branch
```

### Important Notes:

- Avoid working directly on the main branch.
- Do not push changes to the main branch.
- Use pull requests to integrate changes.
- If you have any questions or notice any mistakes, please let me know. 
