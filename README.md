# Vacant_Lot_Detection

## Setup 

1. Set up [Google ADC](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment)
2. Enable the Earth Engine Api through the UI or CLI
    - UI
        ```
        https://console.cloud.google.com/apis/api/earthengine.googleapis.com/metrics?project=<PROJECT>
        ```
    - CLI
        ```
        gcloud services enable earthengine.googleapis.com --project=<PROJECT>
        ```
2. configure roles to Service Usage Consumer + Earth Engine Resource Admin in project https://console.cloud.google.com/iam-admin/iam?project={PROJECT_ID}
3. enabke the earth engine api https://console.cloud.google.com/apis/api/earthengine.googleapis.com/metrics?project=vacant-lot-detection 
4. make sure project is added to gee 
5. register the project https://console.cloud.google.com/earth-engine/configuration;success=true?project=vacant-lot-detection
6. uv run earthengine authenticate --force 
refresh credentials 
