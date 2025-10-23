param(
    [string]$DataPath = "./yolo_ready",
    [string]$Model = "yolov11m.pt",
    [int]$Epochs = 100,
    [int]$Batch = 8,
    [int]$ImgSz = 640,
    [string]$Project = "./runs",
    [string]$Name = "yolo_v11_95_percent",
    [switch]$AutoCreate,
    [int]$Accumulate = 1,
    [string]$WeightsUrl = ''
)

Write-Host "Starting training helper..."

# Activate conda environment
Write-Host "Activate your conda environment first: conda activate yolov11-train"
Write-Host "Or run: conda env create -f environment.yml; conda activate yolov11-train"

if (!(Test-Path $DataPath)) {
    Write-Host "Dataset path not found: $DataPath" -ForegroundColor Red
    exit 1
}

$auto = $AutoCreate.IsPresent ? '--auto-create' : ''
$acc = $Accumulate -gt 1 ? "--accumulate $Accumulate" : ''
$wurl = if ($WeightsUrl -ne '') { "--weights-url `"$WeightsUrl`"" } else { '' }

$cmd = "python .\train_yolov11.py --data `"$DataPath`" --model $Model --epochs $Epochs --batch $Batch --imgsz $ImgSz --project `"$Project`" --name `"$Name`" $auto $acc $wurl --export-onnx"
Write-Host "Running: $cmd"
iex $cmd
