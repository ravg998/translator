pipeline {
    agent any 

    parameters{
        booleanParam(
            name: 'NO_CACHE',
            defaultValue: false, 
            description: "Build Docker without cache is set to True (slower)"
        )
    }
    stages {
        stage("Build docker"){ 
            steps{
                sh "docker build ${params.NO_CACHE ? '--no-cache' : ''} -t translator ."
            }
        }

        stage("Run Translator Tests"){
            steps{
                sh "docker run --rm translator uv run -m pytest"
            }
        }
    }
}