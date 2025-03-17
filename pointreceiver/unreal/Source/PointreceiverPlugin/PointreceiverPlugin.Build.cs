// Copyright Epic Games, Inc. All Rights Reserved.

using System.Diagnostics;
using System.IO;
using UnrealBuildTool;

public class PointreceiverPlugin : ModuleRules
{
	public PointreceiverPlugin(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
		
		// ... add public include paths required here ...
		PublicIncludePaths.AddRange( new string[] { });
		
		// ... add other private include paths required here ...
		PrivateIncludePaths.AddRange( new string[] { });

		// ... add other public dependencies that you statically link with here ...
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "LiveLink", "LiveLinkInterface" });


		// ... add private dependencies that you statically link with here ...	
		PrivateDependencyModuleNames.AddRange(new string[] {
				"CoreUObject",
				"Engine",
				"LiveLink",
				"LiveLinkInterface",
				"Networking",
				"Slate",
				"SlateCore",
				"DeveloperSettings"
			});


		// ... add any modules that your module loads dynamically here ...
		DynamicallyLoadedModuleNames.AddRange( new string[] { });

		// set up pointreceiver library
		PublicIncludePaths.Add(Path.Combine(PluginDirectory, "Library"));
		PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Library", "pointreceiver.lib"));
		PublicDelayLoadDLLs.Add("pointreceiver.dll");
		RuntimeDependencies.Add("$(TargetOutputDir)/pointreceiver.dll",
			Path.Combine(PluginDirectory, "Library", "pointreceiver.dll"));


	}
}
