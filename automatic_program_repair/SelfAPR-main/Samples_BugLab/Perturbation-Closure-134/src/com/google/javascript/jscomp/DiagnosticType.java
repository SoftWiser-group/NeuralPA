[BugLab_Variable_Misuse]^this.defaultLevel = defaultLevel;^96^^^^^94^100^this.defaultLevel = level;^[CLASS] DiagnosticType  [METHOD] <init> [RETURN_TYPE] MessageFormat)   String key CheckLevel level MessageFormat format [VARIABLES] String  key  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^this.level = level;^99^^^^^94^100^this.level = this.defaultLevel;^[CLASS] DiagnosticType  [METHOD] <init> [RETURN_TYPE] MessageFormat)   String key CheckLevel level MessageFormat format [VARIABLES] String  key  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^return make ( key, CheckLevel.ERROR, descriptionFormat ) ;^51^^^^^50^52^return make ( name, CheckLevel.ERROR, descriptionFormat ) ;^[CLASS] DiagnosticType  [METHOD] error [RETURN_TYPE] DiagnosticType   String name String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^return make ( name, CheckLevel.ERROR, key ) ;^51^^^^^50^52^return make ( name, CheckLevel.ERROR, descriptionFormat ) ;^[CLASS] DiagnosticType  [METHOD] error [RETURN_TYPE] DiagnosticType   String name String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^return make ( descriptionFormat, CheckLevel.ERROR, name ) ;^51^^^^^50^52^return make ( name, CheckLevel.ERROR, descriptionFormat ) ;^[CLASS] DiagnosticType  [METHOD] error [RETURN_TYPE] DiagnosticType   String name String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^return make ( key, CheckLevel.WARNING, descriptionFormat ) ;^62^^^^^61^63^return make ( name, CheckLevel.WARNING, descriptionFormat ) ;^[CLASS] DiagnosticType  [METHOD] warning [RETURN_TYPE] DiagnosticType   String name String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^return make ( name, CheckLevel.WARNING, key ) ;^62^^^^^61^63^return make ( name, CheckLevel.WARNING, descriptionFormat ) ;^[CLASS] DiagnosticType  [METHOD] warning [RETURN_TYPE] DiagnosticType   String name String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^return make ( descriptionFormat, CheckLevel.WARNING, name ) ;^62^^^^^61^63^return make ( name, CheckLevel.WARNING, descriptionFormat ) ;^[CLASS] DiagnosticType  [METHOD] warning [RETURN_TYPE] DiagnosticType   String name String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^return make ( key, CheckLevel.OFF, descriptionFormat ) ;^74^^^^^72^75^return make ( name, CheckLevel.OFF, descriptionFormat ) ;^[CLASS] DiagnosticType  [METHOD] disabled [RETURN_TYPE] DiagnosticType   String name String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^return make ( name, CheckLevel.OFF, key ) ;^74^^^^^72^75^return make ( name, CheckLevel.OFF, descriptionFormat ) ;^[CLASS] DiagnosticType  [METHOD] disabled [RETURN_TYPE] DiagnosticType   String name String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^return make ( descriptionFormat, CheckLevel.OFF, name ) ;^74^^^^^72^75^return make ( name, CheckLevel.OFF, descriptionFormat ) ;^[CLASS] DiagnosticType  [METHOD] disabled [RETURN_TYPE] DiagnosticType   String name String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^return new DiagnosticType ( key, level, new MessageFormat ( descriptionFormat )  ) ;^87^88^^^^85^89^return new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^return new DiagnosticType ( name, defaultLevel, new MessageFormat ( descriptionFormat )  ) ;^87^88^^^^85^89^return new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^return new DiagnosticType ( name, level, new MessageFormat ( name )  ) ;^87^88^^^^85^89^return new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^return new DiagnosticType ( level, name, new MessageFormat ( descriptionFormat )  ) ;^87^88^^^^85^89^return new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^return new DiagnosticType ( name, descriptionFormat, new MessageFormat ( level )  ) ;^87^88^^^^85^89^return new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^return new DiagnosticType ( descriptionFormat, level, new MessageFormat ( name )  ) ;^87^88^^^^85^89^return new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^new DiagnosticType ( key, level, new MessageFormat ( descriptionFormat )  ) ;^88^^^^^85^89^new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^new DiagnosticType ( name, level, new MessageFormat ( key )  ) ;^88^^^^^85^89^new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^new DiagnosticType ( descriptionFormat, level, new MessageFormat ( name )  ) ;^88^^^^^85^89^new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^new DiagnosticType ( name, descriptionFormat, new MessageFormat ( level )  ) ;^88^^^^^85^89^new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Variable_Misuse]^new DiagnosticType ( name, level, new MessageFormat ( name )  ) ;^88^^^^^85^89^new DiagnosticType ( name, level, new MessageFormat ( descriptionFormat )  ) ;^[CLASS] DiagnosticType  [METHOD] make [RETURN_TYPE] DiagnosticType   String name CheckLevel level String descriptionFormat [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^return arguments.format ( format ) ;^107^^^^^106^108^return format.format ( arguments ) ;^[CLASS] DiagnosticType  [METHOD] format [RETURN_TYPE] String    arguments [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  Object[]  arguments  
[BugLab_Variable_Misuse]^return name.compareTo ( diagnosticType.key ) ;^112^^^^^111^113^return key.compareTo ( diagnosticType.key ) ;^[CLASS] DiagnosticType  [METHOD] compareTo [RETURN_TYPE] int   DiagnosticType diagnosticType [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  DiagnosticType  diagnosticType  
[BugLab_Variable_Misuse]^return key.compareTo ( name ) ;^112^^^^^111^113^return key.compareTo ( diagnosticType.key ) ;^[CLASS] DiagnosticType  [METHOD] compareTo [RETURN_TYPE] int   DiagnosticType diagnosticType [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  DiagnosticType  diagnosticType  
[BugLab_Argument_Swapping]^return key.compareTo ( diagnosticType.key.key ) ;^112^^^^^111^113^return key.compareTo ( diagnosticType.key ) ;^[CLASS] DiagnosticType  [METHOD] compareTo [RETURN_TYPE] int   DiagnosticType diagnosticType [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  DiagnosticType  diagnosticType  
[BugLab_Argument_Swapping]^return diagnosticType.key.compareTo ( key ) ;^112^^^^^111^113^return key.compareTo ( diagnosticType.key ) ;^[CLASS] DiagnosticType  [METHOD] compareTo [RETURN_TYPE] int   DiagnosticType diagnosticType [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  DiagnosticType  diagnosticType  
[BugLab_Argument_Swapping]^return diagnosticType.compareTo ( key.key ) ;^112^^^^^111^113^return key.compareTo ( diagnosticType.key ) ;^[CLASS] DiagnosticType  [METHOD] compareTo [RETURN_TYPE] int   DiagnosticType diagnosticType [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  DiagnosticType  diagnosticType  
[BugLab_Variable_Misuse]^return name + ": " + format.toPattern (  ) ;^117^^^^^116^118^return key + ": " + format.toPattern (  ) ;^[CLASS] DiagnosticType  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Argument_Swapping]^return format + ": " + key.toPattern (  ) ;^117^^^^^116^118^return key + ": " + format.toPattern (  ) ;^[CLASS] DiagnosticType  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Wrong_Operator]^return key + ": " + format.toPattern (   instanceof   ) ;^117^^^^^116^118^return key + ": " + format.toPattern (  ) ;^[CLASS] DiagnosticType  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
[BugLab_Wrong_Operator]^return key  &  ": " + format.toPattern (  ) ;^117^^^^^116^118^return key + ": " + format.toPattern (  ) ;^[CLASS] DiagnosticType  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] String  descriptionFormat  key  name  boolean  MessageFormat  format  CheckLevel  defaultLevel  level  
