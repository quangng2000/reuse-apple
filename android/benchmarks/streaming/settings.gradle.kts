pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google { content { excludeGroup("io.github.quangng2000") } }
        mavenCentral()
    }
}
rootProject.name = "ReuseStreamingEmulatorBenchmark"
include(":app")

// Explicit development-only substitution; the default remains published0.1.2.
if (providers.gradleProperty("streamingSdkSource").orNull == "local") {
    includeBuild("../..") {
        dependencySubstitution {
            substitute(module("io.github.quangng2000:reuse-streaming-android"))
                .using(project(":reuse-streaming"))
        }
    }
}
