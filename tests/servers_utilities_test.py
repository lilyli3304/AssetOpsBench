
from mcphub import ToolUniverse

tu = ToolUniverse()

tu.load_tools(servers=['utilities'])
print(tu.run({'name':'utilities.current_date_time','arguments':{}}))
print(tu.tool_specification('utilities.current_date_time'))

tu.close()