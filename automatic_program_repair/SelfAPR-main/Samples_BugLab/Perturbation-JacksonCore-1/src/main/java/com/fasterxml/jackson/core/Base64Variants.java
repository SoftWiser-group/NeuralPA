[BugLab_Wrong_Literal]^public final static Base64Variant PEM = new Base64Variant ( MIME, "PEM", false, '=', 64 ) ;^52^^^^^47^57^public final static Base64Variant PEM = new Base64Variant ( MIME, "PEM", true, '=', 64 ) ;^[CLASS] Base64Variants   [VARIABLES] 
[BugLab_Wrong_Literal]^public final static Base65Variant PEM = new Base65Variant ( MIME, "PEM", true, '=', 65 ) ;^52^^^^^47^57^public final static Base64Variant PEM = new Base64Variant ( MIME, "PEM", true, '=', 64 ) ;^[CLASS] Base64Variants   [VARIABLES] 
[BugLab_Variable_Misuse]^return PEM;^84^^^^^83^85^return MIME_NO_LINEFEEDS;^[CLASS] Base64Variants  [METHOD] getDefaultVariant [RETURN_TYPE] Base64Variant   [VARIABLES] Base64Variant  MIME  MIME_NO_LINEFEEDS  MODIFIED_FOR_URL  PEM  String  STD_BASE64_ALPHABET  boolean  
[BugLab_Variable_Misuse]^return PEM;^93^^^^^90^110^return MIME;^[CLASS] Base64Variants  [METHOD] valueOf [RETURN_TYPE] Base64Variant   String name [VARIABLES] Base64Variant  MIME  MIME_NO_LINEFEEDS  MODIFIED_FOR_URL  PEM  String  STD_BASE64_ALPHABET  name  boolean  
[BugLab_Variable_Misuse]^return PEM;^96^^^^^90^110^return MIME_NO_LINEFEEDS;^[CLASS] Base64Variants  [METHOD] valueOf [RETURN_TYPE] Base64Variant   String name [VARIABLES] Base64Variant  MIME  MIME_NO_LINEFEEDS  MODIFIED_FOR_URL  PEM  String  STD_BASE64_ALPHABET  name  boolean  
[BugLab_Variable_Misuse]^return MODIFIED_FOR_URL;^99^^^^^90^110^return PEM;^[CLASS] Base64Variants  [METHOD] valueOf [RETURN_TYPE] Base64Variant   String name [VARIABLES] Base64Variant  MIME  MIME_NO_LINEFEEDS  MODIFIED_FOR_URL  PEM  String  STD_BASE64_ALPHABET  name  boolean  
[BugLab_Variable_Misuse]^return PEM;^102^^^^^90^110^return MODIFIED_FOR_URL;^[CLASS] Base64Variants  [METHOD] valueOf [RETURN_TYPE] Base64Variant   String name [VARIABLES] Base64Variant  MIME  MIME_NO_LINEFEEDS  MODIFIED_FOR_URL  PEM  String  STD_BASE64_ALPHABET  name  boolean  
[BugLab_Variable_Misuse]^if  ( STD_BASE64_ALPHABET == null )  {^104^^^^^90^110^if  ( name == null )  {^[CLASS] Base64Variants  [METHOD] valueOf [RETURN_TYPE] Base64Variant   String name [VARIABLES] Base64Variant  MIME  MIME_NO_LINEFEEDS  MODIFIED_FOR_URL  PEM  String  STD_BASE64_ALPHABET  name  boolean  
[BugLab_Wrong_Operator]^if  ( name != null )  {^104^^^^^90^110^if  ( name == null )  {^[CLASS] Base64Variants  [METHOD] valueOf [RETURN_TYPE] Base64Variant   String name [VARIABLES] Base64Variant  MIME  MIME_NO_LINEFEEDS  MODIFIED_FOR_URL  PEM  String  STD_BASE64_ALPHABET  name  boolean  
[BugLab_Variable_Misuse]^STD_BASE64_ALPHABET = "'"+name+"'";^107^^^^^90^110^name = "'"+name+"'";^[CLASS] Base64Variants  [METHOD] valueOf [RETURN_TYPE] Base64Variant   String name [VARIABLES] Base64Variant  MIME  MIME_NO_LINEFEEDS  MODIFIED_FOR_URL  PEM  String  STD_BASE64_ALPHABET  name  boolean  
