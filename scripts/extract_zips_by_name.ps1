param(
    [string]$RootPath = "",
    [string]$OutputRoot = "",
    [int]$MaxFiles = 0,
    [switch]$NoRecurse,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Find-7Zip {
    $cmd = Get-Command "7z.exe" -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "$env:ProgramFiles\7-Zip\7z.exe",
        "${env:ProgramFiles(x86)}\7-Zip\7z.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }
    return ""
}

function Get-SafeFolderName {
    param([string]$Name)
    $invalid = [System.IO.Path]::GetInvalidFileNameChars()
    $chars = $Name.ToCharArray() | ForEach-Object {
        if ($invalid -contains $_) { "_" } else { $_ }
    }
    return (-join $chars).Trim()
}

function Get-DestinationPath {
    param(
        [System.IO.FileInfo]$ZipFile,
        [string]$Root,
        [string]$OutRoot
    )

    $folderName = Get-SafeFolderName -Name $ZipFile.BaseName
    if ([string]::IsNullOrWhiteSpace($OutRoot)) {
        return Join-Path -Path $ZipFile.DirectoryName -ChildPath $folderName
    }

    $rootFull = [System.IO.Path]::GetFullPath($Root)
    $zipDirFull = [System.IO.Path]::GetFullPath($ZipFile.DirectoryName)
    $relativeDir = ""
    if ($zipDirFull.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relativeDir = $zipDirFull.Substring($rootFull.Length).TrimStart("\", "/")
    }

    if ([string]::IsNullOrWhiteSpace($relativeDir)) {
        return Join-Path -Path $OutRoot -ChildPath $folderName
    }
    return Join-Path -Path (Join-Path -Path $OutRoot -ChildPath $relativeDir) -ChildPath $folderName
}

if ([string]::IsNullOrWhiteSpace($RootPath)) {
    throw "RootPath is required. Example: powershell -ExecutionPolicy Bypass -File .\scripts\extract_zips_by_name.ps1 -RootPath '<zip root path>'"
}

if (-not (Test-Path -LiteralPath $RootPath)) {
    throw "RootPath does not exist: $RootPath"
}

$rootItem = Get-Item -LiteralPath $RootPath
if (-not $rootItem.PSIsContainer) {
    throw "RootPath must be a directory: $RootPath"
}

if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) {
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
}

$sevenZip = Find-7Zip
$searchOption = if ($NoRecurse) {
    [System.IO.SearchOption]::TopDirectoryOnly
} else {
    [System.IO.SearchOption]::AllDirectories
}

$logDir = Join-Path -Path (Get-Location) -ChildPath "artifacts\zip_extract_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path -Path $logDir -ChildPath ("zip_extract_{0:yyyyMMdd_HHmmss}.csv" -f (Get-Date))
$rows = New-Object System.Collections.Generic.List[object]

Write-Host "RootPath : $RootPath"
Write-Host "OutputRoot: $OutputRoot"
Write-Host "ZIP count: streaming search"
Write-Host "MaxFiles : $MaxFiles"
Write-Host "Extractor: $(if ($sevenZip) { $sevenZip } else { 'Expand-Archive' })"
Write-Host "DryRun   : $DryRun"
Write-Host ""

$index = 0
foreach ($zipPath in [System.IO.Directory]::EnumerateFiles($RootPath, "*.zip", $searchOption)) {
    $zip = Get-Item -LiteralPath $zipPath
    $index++
    $dest = Get-DestinationPath -ZipFile $zip -Root $RootPath -OutRoot $OutputRoot
    $status = "Pending"
    $message = ""

    try {
        $exists = Test-Path -LiteralPath $dest
        $hasContent = $false
        if ($exists) {
            $hasContent = [bool](Get-ChildItem -LiteralPath $dest -Force -ErrorAction SilentlyContinue | Select-Object -First 1)
        }

        if ($exists -and $hasContent -and -not $Force) {
            $status = "Skipped"
            $message = "Destination already has files. Use -Force to overwrite."
            Write-Host ("[{0}] SKIP {1} -> {2} | {3}" -f $index, $zip.FullName, $dest, $message)
        } else {
            Write-Progress -Activity "Extract ZIP files" -Status "$index found: $($zip.Name)"
            Write-Host ("[{0}] {1} -> {2}" -f $index, $zip.FullName, $dest)

            if ($DryRun) {
                $status = "DryRun"
                $message = "Not extracted."
            } else {
                New-Item -ItemType Directory -Path $dest -Force | Out-Null
                if ($sevenZip) {
                    $args = @("x", "-y", "-o$dest", $zip.FullName)
                    $proc = Start-Process -FilePath $sevenZip -ArgumentList $args -NoNewWindow -Wait -PassThru
                    if ($proc.ExitCode -ne 0) {
                        throw "7z failed with exit code $($proc.ExitCode)"
                    }
                } else {
                    Expand-Archive -LiteralPath $zip.FullName -DestinationPath $dest -Force:$Force
                }
                $status = "Extracted"
                $message = "OK"
            }
        }
    } catch {
        $status = "Failed"
        $message = $_.Exception.Message
        Write-Warning ("Failed: {0} | {1}" -f $zip.FullName, $message)
    }

    $rows.Add([pscustomobject]@{
        index = $index
        status = $status
        zip_path = $zip.FullName
        destination = $dest
        size_mb = [Math]::Round($zip.Length / 1MB, 2)
        message = $message
    })

    if ($MaxFiles -gt 0 -and $index -ge $MaxFiles) {
        Write-Host "MaxFiles reached: $MaxFiles"
        break
    }
}

Write-Progress -Activity "Extract ZIP files" -Completed
$rows | Export-Csv -Path $logPath -NoTypeInformation -Encoding UTF8

$summary = $rows | Group-Object status | Sort-Object Name | ForEach-Object { "{0}={1}" -f $_.Name, $_.Count }
Write-Host ""
Write-Host "Done: $($summary -join ', ')"
Write-Host "Log : $logPath"
