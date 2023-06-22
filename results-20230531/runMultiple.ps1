<###############################################################################
## User input starts here
###############################################################################>
$sceneDir = 'scenes/'
$scene = 
$outDirAppend = '-pictures'
$spp = 128
$threads = 20
#$maxprimsinnode = 32
$innerCost = '0.5'
$reevaluate = $false
$matchSimdToTree = $false
$useOptimalEpo = $false
$scenes = @(
    'crown'
    'sanmiguel'
    'hairball'
    'living-room'
    'sssdragon'
    'villa'
)
$treeWidths = @(
    2
#    4
#    8
#    16
)
$simdWidths = @(
    1
#    4
#    8
#    16
)
$creationMethods = @(
    'direct'
#    'frombvh'
)
<#Collpase
$optimalEpoRatios = @{
    'sanmiguel' = @{
        4 = '0.4'
        8 = '0.7'
        16= '0.7'
    }
    'hairball' = @{
        4 = '0.2'
        8 = '0.4'
        16= '0.4'
    }
    'crown' = @{
        4 = '0.3'
        8 = '0.4'
        16= '0.5'
    }
    'villa' = @{
        4 = '0.1'
        8 = '0.1'
        16= '0.1'
    }
    'sssdragon' = @{
        4 = '0.5'
        8 = '0.1'
        16= '0.1'
    }
    'living-room' = @{
        4 = '0.2'
        8 = '0.4'
        16= '0.2'
    }
}
#>
#Direct
$optimalEpoRatios = @{
    'sanmiguel' = @{
        2 = '0.1'
        4 = '0.3'
        8 = '0.8'
        16= '0.6'
    }
    'hairball' = @{
        2 = '0.1'
        4 = '0.5'
        8 = '0.2'
        16= '0.3'
    }
    'crown' = @{
        2 = '0.1'
        4 = '0.3'
        8 = '0.4'
        16= '0.4'
    }
    'villa' = @{
        2 = '0.1'
        4 = '0.4'
        8 = '0.6'
        16= '0.5'
    }
    
    'sssdragon' = @{
        2 = '0.1'
        4 = '0.5'
        8 = '0.8'
        16= '0.3'
    }
    'living-room' = @{
        2 = '0.2'
        4 = '0.1'
        8 = '0.1'
        16= '0.1'
    }
}
#>
$epoRatios = @(
    '0.1'
    '0.2'
    '0.3'
    '0.4'
    '0.5'
    '0.6'
    '0.7'
    '0.8'
    '0.9'
)
$optimizations = @(
    'none'
#    'mergeInnerChildren'
#    'mergeIntoParent'
#    'mergeLeaves'
#    'all'
)
$collapseVariants = @(
    0
#    1
#    2
)
$splitVariants = @(
    0
    1
    2
    3
)
$splitMethhods = @(
    'sah'
#    'epo'
)
<###############################################################################
## User input ends here
###############################################################################>
$arguments = @('--stats','--quiet')#'--pixelstats'
$arguments += ('--spp=' + $spp)
$arguments += ('--nthreads=' + $threads)
#$arguments += ('--maxnodeprims=' + $maxprimsinnode)
$arguments += ('--relativeInnerCost=' + $innerCost)
$exeFile = 'pbrt.exe'
Set-Location 'out'
foreach($scene in $scenes){
    $sceneFile = '../../'+$sceneDir + $scene + '/' + $scene + '.pbrt'
    #$outDir = $scene + '-' + $spp + 'spp-' +$maxprimsinnode+ 'prims-innerCost'+$innerCost+'-'+ $outDirAppend +'/'
    $outDir = $scene + '-' + $spp + 'spp-innerCost'+$innerCost+ $outDirAppend +'/'
    New-Item -Path $outDir -ItemType Directory
    Set-Location $outDir
    foreach($treeWidth in $treeWidths)
    {
        $argsWidth = $arguments
        $argsWidth += '--treeWidth='+$treeWidth
        $idWidth = ''+ $treeWidth + 'wide'
        if($matchSimdToTree){
            $usedSimd = @( $treeWidth)
        } else {
            $usedSimd = $simdWidths
        }
        foreach ($simdWidth in $usedSimd){
            $argsSimd = $argsWidth
            $argsSimd += '--simdWidth='+$simdWidth
            $idSimd = $idWidth + '-' + $simdWidth +'simd'
            if($treeWidth -eq 2){
                $usableOptimizations = @('none')
            } else {
                $usableOptimizations = $optimizations
            }
            foreach($optimization in $usableOptimizations){
                $argsopti = $argsSimd
                $argsopti += '--optimizationStrategy='+$optimization
                $idopti = $idSimd + '-' + $optimization
                foreach ($splitMethod in $splitMethhods) {
                    $argsmethod = $argsopti
                    $argsmethod += '--splitMethod='+$splitMethod
                    $idmethod = $idopti + '-' + $splitMethod
                    if($splitMethod -eq 'epo'){
                        if($useOptimalEpo){
                            $ratios = @($optimalEpoRatios[$scene][$treeWidth])
                        }else{
                            $ratios = $eporatios
                        }
                        
                    }else{
                        $ratios = @('0')
                    }
                    foreach($epoRatio in $ratios){
                        $argsratio = $argsmethod
                        $argsratio += '--epoRatio='+$epoRatio
                        $idratio = $idmethod +'-'+$epoRatio+'epoRat'
                        if($treeWidth -eq 2){
                            $usableCreationMethods = @('direct')
                        } else {
                            $usableCreationMethods = $creationMethods
                        }
                        foreach($creationMethod in $usableCreationMethods) {
                            $argscreation = $argsratio
                            $argscreation += '--creationMethod='+$creationMethod
                            $idcreation = $idratio + '-' + $creationMethod
                            if($creationMethod -eq 'direct'){
                                if($treeWidth -eq 2){
                                    $usableSplitVariants = @(0)
                                } else {
                                    $usableSplitVariants = $splitVariants
                                }
                                foreach($splitVariant in $usableSplitVariants) {
                                    $args = $argscreation
                                    $args += '--splitVariant='+$splitVariant
                                    $id = $idcreation + '-' + $splitVariant
                                    $outFile = $id + '.out'
                                    $imgFile = $id + '.exr'
                                    $outParam = '--outfile=' + $imgFile
                                    if((Test-Path -LiteralPath $imgFile) -and (-not($reevaluate)))
                                    {
                                        Write-Output ('Skipped ' + $id)
                                        continue
                                    }
                                    $logging = '--log-file=' + $id + '.log'
                                    $args += $logging
                                    $args += $outParam
                                    $args += $sceneFile
                                    Write-Output ('Started ' + $id)
                                    Write-Output 'executing'+ $exeFile +$args > $outfile
                                    & $exeFile $args 2>&1 >> $outFile
                                    Write-Output 'Finished '
                                }
                            } else {
                                foreach($collapseVariant in $collapseVariants) {
                                    $args = $argscreation
                                    $args += '--collapseVariant='+$collapseVariant
                                    $id = $idcreation + '-' + $collapseVariant
                                    $outFile = $id + '.out'
                                    $imgFile = $id + '.exr'
                                    $outParam = '--outfile=' + $imgFile
                                    if((Test-Path -LiteralPath $imgFile) -and (-not($reevaluate)))
                                    {
                                        Write-Output ('Skipped ' + $id)
                                        continue
                                    }
                                    $logging = '--log-file=' + $id + '.log'
                                    $args += $logging
                                    $args += $outParam
                                    $args += $sceneFile
                                    Write-Output ('Started ' + $id)
                                    Write-Output 'executing'+ $exeFile +$args > $outfile
                                    & $exeFile $args 2>&1 >> $outFile
                                    Write-Output 'Finished '
                                }
                            }
                        
                        }
                    }
                }
            }
        }
    }
    Set-Location '..'
}
Set-Location '..'
Write-Host "Press any key to continue..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
#& shutdown /s /t 0
