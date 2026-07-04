pipeline {
    agent any 
    stages {
        stage("Build docker"){ 
            steps{
                sh "docker build -t translator ."
            }
        }

        stage("Run Translator Tests"){
            steps{
                sh "docker run --rm translator uv run pytest"
            }
        }
    }
}