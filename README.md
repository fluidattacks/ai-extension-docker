# Fluid Attacks AI Docker Container

[Fluid Attacks](https://fluidattacks.com) AI
helps you detect the files in your commits
that are more likely to contain security vulnerabilities,
enabling you to prioritize their review.

This repository contains a docker container
of the Fluid Attacks AI so you can use it in your projects.

## Getting started

This document contains the necessary steps
to use the AI in your project's pipeline.

### Requirements

- In order to use the Fluid Attacks AI for your project
  you only need to have its git repository
  in one of the following platforms:
  - [GitLab](https://gitlab.com/)
  - [GitHub](https://github.com/)
  - [Bitbucket](https://bitbucket.org/)
  - [CircleCI](https://circleci.com/)
  - [Jenkins](https://www.jenkins.io/)

### Adding the Fluid Attacks AI to your pipeline

As a docker containerized application
you should theoretically be able
to utilize the Fluid Attacks AI
in any platform where docker images
can be used,
you only need to have access
to your git repository
and use the following command:

`$ sorts git_repository_path break_pipeline commit_risk_limit`

The command needs the three following arguments:

- **git_repository_path:**
  This is the path inside the container
  where your git repository is located.
  This path differs depending on the platform used,
  plase consult the platform's documentation
  in order to know the path
  or the environment variable for it.
- **break_pipeline:**
  This argument determines
  if your pipeline should break
  in case the median commit risk detected
  surpasses a threshold that you specify.
  This argument needs to be a boolean (`True` or `False`).
- **commit_risk_limit:**
  This is a percentage number
  that defines the threshold
  that will cause your pipeline to break
  in case you set the `break_pipeline` to `True`.

This command will remain the same across platforms,
however each platform has a different way of writing
the file that defines your project's pipeline.
Coming up next are examples of files
for configurating jobs in several platforms
where the Fluid Attacks AI
has already been tested
and confirmed to work properly.

#### Gitlab

```yaml
ai_job_name:
  image: ghcr.io/fluidattacks/sorts-extension:latest
  script:
    - sorts $PWD True 75
```

#### GitHub

```yaml
jobs:
  ai_job_name:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/fluidattacks/sorts-extension:latest
    steps:
      - uses: actions/checkout@v1
      - name: Check the commit risk with Sorts
        run: sorts $GITHUB_WORKSPACE True 75
```

#### Bitbucket

```yaml
image: ghcr.io/fluidattacks/sorts-extension:latest
   
pipelines:
  default:
    - step:
        name: Check the commit risk with Sorts
        script:
          - sorts $BITBUCKET_CLONE_DIR True 75
```

#### CircleCI

```yaml
version: 2.1

jobs:
  ai_job_name:
    docker:
      - image: ghcr.io/fluidattacks/sorts-extension:latest
    steps:
      - checkout
      - run:
          name: "Check the commit risk with Sorts"
          command: "sorts $CIRCLE_WORKING_DIRECTORY True 75"

workflows:
  ai_job_name_workflow:
    jobs:
      - ai_job_name
```

#### Jenkins

```
pipeline {
    agent {
        docker { image 'ghcr.io/fluidattacks/sorts-extension:latest' }
    }
    stages {
        stage('Check the commit risk with Sorts') {
            steps {
                sh 'sorts $WORKSPACE True 75'
            }
        }
    }
}
```

Depending on the platform used,
you may need to add or modify the file
according to your needs
if you need to trigger the job
under certain conditions
or in specific events.
