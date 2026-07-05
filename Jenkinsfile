pipeline {
    agent any 

    parameters{
        booleanParam(
            name: 'NO_CACHE',
            defaultValue: false, 
            description: "Build Docker without cache is set to True (slower)"
        )
    }

    environment { 
        IMAGE="ghcr.io/ravg998/translator"
        BRANCH_TO_PUSH="origin/main"
        
    }
    stages {
        stage("Build docker"){ 
            steps{
                sh "docker build ${params.NO_CACHE ? '--no-cache' : ''} -t translator:latest ."
            }
        }

        stage("Run Translator Tests"){
            steps{
                sh "docker run --rm translator:latest uv run -m pytest"
            }
        }
        
        stage("Push Image"){ 
            when{
                expression { env.GIT_BRANCH == env.BRANCH_TO_PUSH}
            }
            steps{
                withCredentials([usernamePassword(
                    credentialsId: "git_cred", 
                    usernameVariable: "GHCR_USER", 
                    passwordVariable: "GHCR_PWD"
                )]) { 
                    sh ''' 
                    echo "$GHCR_PWD" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
                    docker tag translator ${IMAGE}:latest
                    docker push ${IMAGE}:latest
                    docker logout ghcr.io
                    '''
                }
            }
        }
        stage("Deploy to VM") {
            when {
                expression { env.GIT_BRANCH == env.BRANCH_TO_PUSH}
            }
            steps {
                sshagent(credentials: ['gcp-ssh-key']) {
                    sh '''
                        scp -o StrictHostKeyChecking=no  docker-compose.yml docker-compose.prod.yml   jenkins-deploy@34.13.251.27:~/translator-deploy/
                            
                        ssh -o StrictHostKeyChecking=no jenkins-deploy@34.13.251.27 "
                            cd ~/translator-deploy/ &&
                            docker stop translator || true &&
                            docker rm translator || true &&
                            docker rmi env.IMAGE:latest || true &&
                            docker image prune -f && 
                            docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
                        "
                    '''
                }
            }
        }
    }
}