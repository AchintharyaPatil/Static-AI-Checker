import json

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def check_compatibility(plugins):
    compatibility_issues = []

    # Create a mapping from source types to their compatible sinks
    source_to_sinks = {}
    for plugin, types in plugins.items():
        for source in types.get('source', []):
            if source not in source_to_sinks:
                source_to_sinks[source] = set()
            source_to_sinks[source].update(types.get('sink', []))

    # Check for compatibility between each pair of plugins
    plugin_list = list(plugins.keys())
    for i in range(len(plugin_list) - 1):
        plugin1 = plugin_list[i]
        types1 = plugins[plugin1]
        sources1 = types1.get('source', [])
        sinks1 = types1.get('sink', [])
        if not sources1 and not sinks1:  # Skip if both sources and sinks are empty
            continue
        
        # Compare with the next plugin
        plugin2 = plugin_list[i + 1]
        types2 = plugins[plugin2]
        sources2 = types2.get('source', [])
        sinks2 = types2.get('sink', [])
        if not sources2 and not sinks2:  # Skip if both sources and sinks are empty
            continue

        # Check compatibility considering empty sources and sinks
        compatible_found = False  # Flag to check if any compatibility is found

        # If either source or sink is empty, consider them compatible with anything else
        if not sources1 or not sinks2 or not sources2 or not sinks1:
            compatible_found = True
        else:
            # Check for each source type of plugin1
            for source1 in sources1:
                # If source1 is "ANY" or starts with "other/", consider it compatible
                if source1 == "ANY" or source1.startswith("other/"):
                    compatible_found = True
                    break

                compatible_sinks = source_to_sinks.get(source1, set())

                # Check if any sink of plugin2 is compatible with the sources of plugin1
                for sink2 in sinks2:
                    # If sink2 is "ANY", consider it compatible
                    if sink2 == "ANY":
                        compatible_found = True
                        break
                    
                    # Check if the sink of plugin2 is a compatible sink for source1
                    if sink2 in compatible_sinks:
                        compatible_found = True  # Found a compatible source or sink
                        break  # No need to check further
                if compatible_found:
                    break  # Exit the outer loop if compatible found

            # If no compatible source-sink pair was found, check the reverse
            if not compatible_found:
                for source2 in sources2:
                    # If source2 is "ANY" or starts with "other/", consider it compatible
                    if source2 == "ANY" or source2.startswith("other/"):
                        compatible_found = True
                        break
                    for sink1 in sinks1:
                        # If sink1 is "ANY", consider it compatible
                        if sink1 == "ANY":
                            compatible_found = True
                            break
                        compatible_sinks = source_to_sinks.get(source2, set())
                        if sink1 in compatible_sinks:
                            compatible_found = True
                            break
                    if compatible_found:
                        break

        # If no compatible source-sink pair was found, add to issues
        if not compatible_found:
            compatibility_issues.append((plugin1, plugin2, sources1))

    return compatibility_issues

def main():
    """Main function to load plugins and check for compatibility issues."""
    plugins = load_json("plugin_compatibility_check/gstreamer_caps.json")
    issues = check_compatibility(plugins)

    if issues:
        for issue in issues:
            print(f"Compatibility issue between {issue[0]} and {issue[1]} for source type(s) {issue[2]}")
        print(f"\nTotal issues: {len(issues)}")
    else:
        print("All source and sink types are compatible.")

if __name__ == "__main__":
    main()
