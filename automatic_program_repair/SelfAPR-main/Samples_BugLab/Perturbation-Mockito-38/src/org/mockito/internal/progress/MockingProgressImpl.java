[BugLab_Variable_Misuse]^return iOngoingStubbing;^34^^^^^31^35^return temp;^[CLASS] MockingProgressImpl  [METHOD] pullOngoingStubbing [RETURN_TYPE] IOngoingStubbing   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  stubbingInProgress  
[BugLab_Wrong_Operator]^if  ( verificationMode != null )  {^51^^^^^50^58^if  ( verificationMode == null )  {^[CLASS] MockingProgressImpl  [METHOD] pullVerificationMode [RETURN_TYPE] VerificationMode   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  VerificationMode  temp  Location  stubbingInProgress  
[BugLab_Wrong_Operator]^if  ( verificationMode == null )  {^70^^^^^65^83^if  ( verificationMode != null )  {^[CLASS] MockingProgressImpl  [METHOD] validateState [RETURN_TYPE] void   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Variable_Misuse]^reporter.unfinishedVerificationException ( temp ) ;^73^^^^^65^83^reporter.unfinishedVerificationException ( location ) ;^[CLASS] MockingProgressImpl  [METHOD] validateState [RETURN_TYPE] void   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Variable_Misuse]^if  ( location != null )  {^76^^^^^65^83^if  ( stubbingInProgress != null )  {^[CLASS] MockingProgressImpl  [METHOD] validateState [RETURN_TYPE] void   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Wrong_Operator]^if  ( stubbingInProgress == null )  {^76^^^^^65^83^if  ( stubbingInProgress != null )  {^[CLASS] MockingProgressImpl  [METHOD] validateState [RETURN_TYPE] void   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Variable_Misuse]^Location temp = location;^77^^^^^65^83^Location temp = stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] validateState [RETURN_TYPE] void   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Variable_Misuse]^reporter.unfinishedStubbing ( stubbingInProgress ) ;^79^^^^^65^83^reporter.unfinishedStubbing ( temp ) ;^[CLASS] MockingProgressImpl  [METHOD] validateState [RETURN_TYPE] void   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Variable_Misuse]^return  "iOngoingStubbing: " + temp + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^91^92^93^^^90^94^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Variable_Misuse]^return  "iOngoingStubbing: " + iOngoingStubbing + ", null: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^91^92^93^^^90^94^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Variable_Misuse]^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", temp: " + stubbingInProgress;^91^92^93^^^90^94^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Argument_Swapping]^return  "iOngoingStubbing: " + stubbingInProgress + ", verificationMode: " + verificationMode + ", iOngoingStubbing: " + stubbingInProgress;^91^92^93^^^90^94^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Argument_Swapping]^return  "iOngoingStubbing: " + iOngoingStubbing + ", stubbingInProgress: " + verificationMode + ", verificationMode: " + stubbingInProgress;^91^92^93^^^90^94^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Wrong_Operator]^return  !=  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^91^92^93^^^90^94^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Wrong_Operator]^return  ||  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^91^92^93^^^90^94^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Wrong_Operator]^return  <=  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^91^92^93^^^90^94^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
[BugLab_Wrong_Operator]^return  "iOngoingStubbing: "  >>  iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^91^92^93^^^90^94^return  "iOngoingStubbing: " + iOngoingStubbing + ", verificationMode: " + verificationMode + ", stubbingInProgress: " + stubbingInProgress;^[CLASS] MockingProgressImpl  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  Reporter  reporter  DebuggingInfo  debuggingInfo  IOngoingStubbing  iOngoingStubbing  temp  ArgumentMatcherStorage  argumentMatcherStorage  Localized  verificationMode  Location  location  stubbingInProgress  temp  
