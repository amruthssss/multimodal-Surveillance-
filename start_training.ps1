param(
    [string]$DataPath = "./yolo_ready_final",
    [string]$Model = "yolov11m.pt",
    [int]$Epochs = 150,
    [int]$Batch = 8,
    [int]$ImgSz = 640,
    [string]$Project = "./runs",
    [string]$Name = "yolo_cctv_4class",
    [switch]$AutoCreate,
    [int]$Accumulate = 4,
    [string]$WeightsUrl = ''
)

Write-Host "🚀 YOLOv11 Training Helper for 4-Class CCTV Detection" -ForegroundColor Cyan
Write-Host "   (fire, vehicle_accident, fighting, explosion)" -ForegroundColor Gray
Write-Host ""

# Activate conda environment
Write-Host "⚙️  Activate your conda environment first:" -ForegroundColor Yellow
Write-Host "   conda activate yolov11-train" -ForegroundColor White
Write-Host "   OR: conda env create -f environment.yml; conda activate yolov11-train" -ForegroundColor Gray
Write-Host ""

if (!(Test-Path $DataPath)) {
    Write-Host "❌ Dataset path not found: $DataPath" -ForegroundColor Red
    Write-Host "   Expected structure: $DataPath/images/{train,val}, $DataPath/labels/{train,val}" -ForegroundColor Gray
    exit 1
}

Write-Host "✅ Dataset found: $DataPath" -ForegroundColor Green

# Check for data.yaml
$dataYaml = Join-Path $DataPath "data.yaml"
if (Test-Path $dataYaml) {
    Write-Host "✅ data.yaml found" -ForegroundColor Green
} elseif ($AutoCreate.IsPresent) {
    Write-Host "⚠️  data.yaml not found - will auto-create" -ForegroundColor Yellow
} else {
    Write-Host "❌ data.yaml not found. Use --AutoCreate to generate it automatically" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📋 Training Configuration:" -ForegroundColor Cyan
Write-Host "   Model: $Model" -ForegroundColor White
Write-Host "   Epochs: $Epochs" -ForegroundColor White
Write-Host "   Batch: $Batch (Accumulate: $Accumulate = Effective batch: $($Batch * $Accumulate))" -ForegroundColor White
Write-Host "   Image Size: $ImgSz" -ForegroundColor White
Write-Host "   Output: $Project/$Name" -ForegroundColor White
Write-Host ""

$auto = $AutoCreate.IsPresent ? '--auto-create' : ''
$acc = $Accumulate -gt 1 ? "--accumulate $Accumulate" : ''
$wurl = if ($WeightsUrl -ne '') { "--weights-url `"$WeightsUrl`"" } else { '' }

$cmd = "python .\train_yolov11.py --data `"$DataPath`" --model $Model --epochs $Epochs --batch $Batch --imgsz $ImgSz --project `"$Project`" --name `"$Name`" $auto $acc $wurl --export-onnx"
Write-Host "🔥 Executing: $cmd" -ForegroundColor Yellow
Write-Host ""
iex $cmd
