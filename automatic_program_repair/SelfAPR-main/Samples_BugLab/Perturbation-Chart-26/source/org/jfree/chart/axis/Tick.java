[BugLab_Variable_Misuse]^if  ( rotationAnchor == null )  {^87^^^^^85^99^if  ( textAnchor == null )  {^[CLASS] Tick  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   String text TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( textAnchor != null )  {^87^^^^^85^99^if  ( textAnchor == null )  {^[CLASS] Tick  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   String text TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( textAnchor == null )  {^90^^^^^85^99^if  ( rotationAnchor == null )  {^[CLASS] Tick  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   String text TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( rotationAnchor != null )  {^90^^^^^85^99^if  ( rotationAnchor == null )  {^[CLASS] Tick  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   String text TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^this.textAnchor = rotationAnchor;^96^^^^^85^99^this.textAnchor = textAnchor;^[CLASS] Tick  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   String text TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^this.rotationAnchor = textAnchor;^97^^^^^85^99^this.rotationAnchor = rotationAnchor;^[CLASS] Tick  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   String text TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^return text;^107^^^^^106^108^return this.text;^[CLASS] Tick  [METHOD] getText [RETURN_TYPE] String   [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^return textAnchor;^126^^^^^125^127^return this.rotationAnchor;^[CLASS] Tick  [METHOD] getRotationAnchor [RETURN_TYPE] TextAnchor   [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^return angle;^135^^^^^134^136^return this.angle;^[CLASS] Tick  [METHOD] getAngle [RETURN_TYPE] double   [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( this <= obj )  {^146^^^^^145^166^if  ( this == obj )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^147^^^^^145^166^return true;^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( obj  >=  Tick )  {^149^^^^^145^166^if  ( obj instanceof Tick )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( text, t.text )  )  {^151^^^^^145^166^if  ( !ObjectUtilities.equal ( this.text, t.text )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( t.texthis.text, t )  )  {^151^^^^^145^166^if  ( !ObjectUtilities.equal ( this.text, t.text )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( t.text, this.text )  )  {^151^^^^^145^166^if  ( !ObjectUtilities.equal ( this.text, t.text )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^152^^^^^145^166^return false;^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( textAnchor, t.textAnchor )  )  {^154^^^^^145^166^if  ( !ObjectUtilities.equal ( this.textAnchor, t.textAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( this.textAnchor, textAnchor )  )  {^154^^^^^145^166^if  ( !ObjectUtilities.equal ( this.textAnchor, t.textAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( this.textAnchorhis.textAnchor, t.textAnchor )  )  {^154^^^^^145^166^if  ( !ObjectUtilities.equal ( this.textAnchor, t.textAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( t.textAnchor, this.textAnchor )  )  {^154^^^^^145^166^if  ( !ObjectUtilities.equal ( this.textAnchor, t.textAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^155^^^^^145^166^return false;^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( textAnchor, t.rotationAnchor )  )  {^157^^^^^145^166^if  ( !ObjectUtilities.equal ( this.rotationAnchor, t.rotationAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( this.rotationAnchor, textAnchor )  )  {^157^^^^^145^166^if  ( !ObjectUtilities.equal ( this.rotationAnchor, t.rotationAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( t.rotationAnchorhis.rotationAnchor, t )  )  {^157^^^^^145^166^if  ( !ObjectUtilities.equal ( this.rotationAnchor, t.rotationAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( t.rotationAnchor, this.rotationAnchor )  )  {^157^^^^^145^166^if  ( !ObjectUtilities.equal ( this.rotationAnchor, t.rotationAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^158^^^^^145^166^return false;^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( ! ( angle == t.angle )  )  {^160^^^^^145^166^if  ( ! ( this.angle == t.angle )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( ! ( this.angle == angle )  )  {^160^^^^^145^166^if  ( ! ( this.angle == t.angle )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( ! ( this.anglehis.angle == t.angle )  )  {^160^^^^^145^166^if  ( ! ( this.angle == t.angle )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( ! ( t.angle == this.angle )  )  {^160^^^^^145^166^if  ( ! ( this.angle == t.angle )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( ! ( this.angle != t.angle )  )  {^160^^^^^145^166^if  ( ! ( this.angle == t.angle )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^161^^^^^145^166^return false;^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^163^^^^^145^166^return true;^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( this.text, text )  )  {^151^^^^^145^166^if  ( !ObjectUtilities.equal ( this.text, t.text )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( this.texthis.text, t.text )  )  {^151^^^^^145^166^if  ( !ObjectUtilities.equal ( this.text, t.text )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( t.textAnchorhis.textAnchor, t )  )  {^154^^^^^145^166^if  ( !ObjectUtilities.equal ( this.textAnchor, t.textAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( this.rotationAnchorhis.rotationAnchor, t.rotationAnchor )  )  {^157^^^^^145^166^if  ( !ObjectUtilities.equal ( this.rotationAnchor, t.rotationAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( t, this.text.text )  )  {^151^^^^^145^166^if  ( !ObjectUtilities.equal ( this.text, t.text )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( t, this.rotationAnchor.rotationAnchor )  )  {^157^^^^^145^166^if  ( !ObjectUtilities.equal ( this.rotationAnchor, t.rotationAnchor )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( ! ( t.anglehis.angle == t )  )  {^160^^^^^145^166^if  ( ! ( this.angle == t.angle )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( ! ( this.angle <= t.angle )  )  {^160^^^^^145^166^if  ( ! ( this.angle == t.angle )  )  {^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^165^^^^^145^166^return false;^[CLASS] Tick  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  double  angle  Object  obj  Tick  t  String  text  long  serialVersionUID  
[BugLab_Variable_Misuse]^return text;^186^^^^^185^187^return this.text;^[CLASS] Tick  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] TextAnchor  rotationAnchor  textAnchor  String  text  boolean  double  angle  long  serialVersionUID  