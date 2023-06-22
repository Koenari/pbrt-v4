import os

def numParser(line):
        return int(line)
def floatParser(line):
        return float(line)
def msParser(line):
        return float(line.split("ms")[0])
def mbParser(line):
        if(line.__contains__("kB")):
                return kbParser(line) /1000
        else:
                return float(line.split("MiB")[0]) 
        
def kbParser(line):
        return float(line.split("kB")[0])
def ratioParser1(line):
        numbers = line.split('(')[0].split('/')
        return int(numbers[0])
def ratioParser2(line):
        numbers = line.split('(')[0].split('/')
        return int(numbers[1])
def distAvgParser(line):
        return float(line.split("avg")[0])
def distMinParser(line):
        return float(line.split("range")[1].split('-')[0])
def distMaxParser(line):
        return float(line.split("-")[1].split(']')[0])
def configParsePre(config, desc):
        return config.split(desc)[-2].split('-')[-1]
def configParsePost(config, desc):
        return config.split(desc)[-1].split('-')[0]
def configParseLast(config, desc):
        return config.split('-')[-1]
def configParseMethod(config,desc):
        if(config.__contains__("frombvh")):
                return "frombvh"
        else:
                return "direct"
def configParseMetric(config,desc):
        if(config.__contains__("sah")):
                return "sah"
        else:
                return "epo"
def configParseOpti(config,desc):
        if(config.__contains__("all")):
                return "all"
        else:
                return "none"
outColumnDefinitions = [
        "Configuration",
        "CreationTime",
        "OptimizationTime",
        "InteriorNodes",
        "LeafNodes",
        "EmptyNodes",
        "PrimCount",
        "RayCount",
        "NodeIntersections",
        "PrimIntersections",
        "Metric",
        "SISDNodeIntersections",
        "SISDPrimIntersections",
        "TreeWidth",
        "SimdWidth",
        "Optimizations",
        "CreationMethod",
        "SplitMethod",
        "Variant",
        "EpoRatio",
        "OptiMergedInner",
        "OptiMergedLeaf",
        "OptiMergedParent",
        "LeafDepthAvg",
        "LeafDepthMin",
        "LeafDepthMax",
        "InteriorDepthAvg",
        "TotalNodes",
]
outNames = {
        "InteriorNodes" : "Count Interior Nodes",
        "LeafNodes" : "Count Primitives per leaf",
        "EmptyNodes" : "Count Empty Nodes",
        "TotalNodes": "Count Empty Nodes",
        "PrimCount" : "Count Primitives per leaf",
        "NodeIntersections" : "SIMD Interior Nodes visited",
        "PrimIntersections" : "SIMD Primitive intersections",
        "SISDNodeIntersections" : "SISD Nodes visited",
        "SISDPrimIntersections" : "SISD Primitive intersections",
        "LeafDepthAvg" : "Depth Leaves",
        "LeafDepthMin" : "Depth Leaves",
        "LeafDepthMax" : "Depth Leaves",
        "InteriorDepthAvg" : "Depth Interior",
        "InteriorDepthMin" : "Depth Interior",
        "InteriorDepthMax" : "Depth Interior",
        "BVHMemory": "BVH Memory",
        "RayCount" : "Camera rays traced",
        "OptiMergedInner" : "Merged Inner",
        "OptiMergedLeaf" : "Merged Leaves",
        "OptiMergedParent" : "Merged Into Parent ",
        }
logNames = {
        "CreationTime" : "Creation took:",
        "OptimizationTime" : "Optimization took:",
        "Metric":"Calulated cost for BVH is:",
        }
parserFunction = {
        "InteriorNodes" : numParser,
        "LeafNodes" : ratioParser2,
        "EmptyNodes" : ratioParser1,
        "TotalNodes" : ratioParser2,
        "PrimCount" : ratioParser1,
        "NodeIntersections" : numParser,
        "PrimIntersections" : numParser,
        "SISDNodeIntersections" : numParser,
        "SISDPrimIntersections" : numParser,
        "Metric" : floatParser,
        "CreationTime" : msParser,
        "OptimizationTime": msParser,
        "BVHMemory": mbParser,
        "LeafDepthAvg": distAvgParser,
        "LeafDepthMin": distMinParser,
        "LeafDepthMax": distMaxParser,
        "InteriorDepthAvg" : distAvgParser,
        "InteriorDepthMin" : distMinParser,
        "InteriorDepthMax" : distMaxParser,
        "RayCount" : numParser,
        "OptiMergedInner" : numParser,
        "OptiMergedLeaf" : numParser,
        "OptiMergedParent" : numParser,
        
        }
configNames = {
        "TreeWidth" : "wide",
        "SimdWidth" : "simd",
        "Variant" : "var",
        "EpoRatio" : "epoRat",
        "Optimizations" : "",
        "SplitMethod" : "",
        "CreationMethod" : "",
}
configFunctions = {
        "TreeWidth" : configParsePre,
        "SimdWidth" : configParsePre,
        "Variant" : configParseLast,
        "EpoRatio" : configParsePre,
        "Optimizations" : configParseOpti,
        "SplitMethod" : configParseMetric,
        "CreationMethod" : configParseMethod,
}
def parseConfiguration(output):
        config = output["Configuration"]
        for key, desc in configNames.items():
                output[key] = configFunctions[key](config, desc)
                
def proccessResult(output, outfile, logfile):
        shouldParse = True
        for line in outfile.readlines():
                for key, statName in outNames.items():
                        if line.__contains__(statName):
                                output[key] = parserFunction[key](line.split(statName)[1])
        shouldParse = False
        for line in logfile.readlines():
                if line.__contains__("Starting top-level accelerator"):
                        shouldParse = True
                elif line.__contains__("Finished top-level accelerator"):
                        break
                if shouldParse:
                        line = line.split(']')[1]
                        for key, statName in logNames.items():
                                if line.__contains__(statName):
                                        output[key] = parserFunction[key](line.split(statName)[1])
        shouldParse = True
        return output
        
def writeHeaderRow(csvfile):
        line = ""
        for col in outColumnDefinitions:
                line = line + col + ','
        csvfile.write(line[:-1]+"\n")
        
def writeRow(csvfile, result):
        line = ""
        for col in outColumnDefinitions:
                if col in result:
                        line = line + str(result[col]) + ','
                else:
                        line = line + ','
        csvfile.write(line[:-1]+"\n")

if __name__ == "__main__":
        inDir = "out"
        outDir = "csv/"
        directories = [dirs for path, dirs, _ in os.walk(inDir) if path == inDir][0]
        for folder in directories:
                fulldir = inDir+'/'+folder+'/'
                print('{}'.format(fulldir))
                files = [file for path, _, file in os.walk(fulldir) if path == fulldir][0]
                outfiles = filter(lambda x : x.endswith(".out"),files)
                with open(outDir+folder+'.csv',"w") as csvfile:
                        writeHeaderRow(csvfile)
                        for outfileName in outfiles:
                                with open(fulldir+outfileName[:-3]+"log","r") as logfile:
                                        with open(fulldir+outfileName,"r", encoding="utf_16") as outfile:
                                                print(fulldir+outfileName)
                                                result = {}
                                                result["Empty"] = ''
                                                result["Configuration"] = outfileName[:-4]
                                                parseConfiguration(result)
                                                proccessResult(result, outfile, logfile)
                                                writeRow(csvfile, result)
                                                #print(repr(result))
                        
