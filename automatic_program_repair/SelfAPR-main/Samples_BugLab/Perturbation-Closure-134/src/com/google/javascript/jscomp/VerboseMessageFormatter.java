[BugLab_Wrong_Operator]^return getLevelName ( CheckLevel.ERROR )  <<  + ": " + format ( error ) ;^33^^^^^32^34^return getLevelName ( CheckLevel.ERROR )  + ": " + format ( error ) ;^[CLASS] VerboseMessageFormatter  [METHOD] formatError [RETURN_TYPE] String   JSError error [VARIABLES] boolean  JSError  error  
[BugLab_Wrong_Operator]^return getLevelName ( CheckLevel.ERROR )   !=  ": " + format ( error ) ;^33^^^^^32^34^return getLevelName ( CheckLevel.ERROR )  + ": " + format ( error ) ;^[CLASS] VerboseMessageFormatter  [METHOD] formatError [RETURN_TYPE] String   JSError error [VARIABLES] boolean  JSError  error  
[BugLab_Wrong_Operator]^return getLevelName ( CheckLevel.WARNING )  ^  + ": " + format ( warning ) ;^37^^^^^36^38^return getLevelName ( CheckLevel.WARNING )  + ": " + format ( warning ) ;^[CLASS] VerboseMessageFormatter  [METHOD] formatWarning [RETURN_TYPE] String   JSError warning [VARIABLES] boolean  JSError  warning  
[BugLab_Wrong_Operator]^return getLevelName ( CheckLevel.WARNING )    instanceof   ": " + format ( warning ) ;^37^^^^^36^38^return getLevelName ( CheckLevel.WARNING )  + ": " + format ( warning ) ;^[CLASS] VerboseMessageFormatter  [METHOD] formatWarning [RETURN_TYPE] String   JSError warning [VARIABLES] boolean  JSError  warning  
[BugLab_Variable_Misuse]^Region sourceRegion = getSource (  ) .getSourceRegion ( lineSource, lineNumber ) ;^44^^^^^40^53^Region sourceRegion = getSource (  ) .getSourceRegion ( sourceName, lineNumber ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Argument_Swapping]^Region sourceRegion = getSource (  ) .getSourceRegion ( lineNumber, sourceName ) ;^44^^^^^40^53^Region sourceRegion = getSource (  ) .getSourceRegion ( sourceName, lineNumber ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Wrong_Operator]^if  ( sourceRegion == null )  {^46^^^^^40^53^if  ( sourceRegion != null )  {^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Variable_Misuse]^return String.format ( "%s at %s line %s %s", lineSource, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Variable_Misuse]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( description )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Variable_Misuse]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( description != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Argument_Swapping]^return String.format ( "%s at %s line %s %s", lineSource, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( description != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Argument_Swapping]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( lineSource )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( sourceName != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Argument_Swapping]^return String.format ( "%s at %s line %s %s", lineNumber, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( description < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Wrong_Operator]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber <= 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Wrong_Operator]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource == null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Wrong_Operator]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n"  >  lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Wrong_Literal]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < lineNumber )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Variable_Misuse]^return String.format ( "%s at %s line %s %s", sourceName, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Variable_Misuse]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( sourceName != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Argument_Swapping]^return String.format ( "%s at %s line %s %s", sourceName, ( StringUtil.isEmpty ( description )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Argument_Swapping]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( lineNumber )  ? " ( unknown source ) " : sourceName ) , (  ( sourceName < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Wrong_Operator]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber > 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Wrong_Operator]^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n"  &  lineSource : "." )  ) ;^49^50^51^52^^40^53^return String.format ( "%s at %s line %s %s", description, ( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  
[BugLab_Variable_Misuse]^( StringUtil.isEmpty ( lineSource )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^50^51^52^^^40^53^( StringUtil.isEmpty ( sourceName )  ? " ( unknown source ) " : sourceName ) , (  ( lineNumber < 0 )  ? String.valueOf ( lineNumber )  : " ( unknown line ) " ) , (  ( lineSource != null )  ? ":\n\n" + lineSource : "." )  ) ;^[CLASS] VerboseMessageFormatter  [METHOD] format [RETURN_TYPE] String   JSError message [VARIABLES] boolean  Region  sourceRegion  String  description  lineSource  sourceName  JSError  message  int  lineNumber  