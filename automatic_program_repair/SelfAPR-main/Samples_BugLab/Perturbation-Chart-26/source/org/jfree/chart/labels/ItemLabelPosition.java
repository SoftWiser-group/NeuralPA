[BugLab_Variable_Misuse]^this ( itemLabelAnchor, rotationAnchor, TextAnchor.CENTER, 0.0 ) ;^92^^^^^90^93^this ( itemLabelAnchor, textAnchor, TextAnchor.CENTER, 0.0 ) ;^[CLASS] ItemLabelPosition  [METHOD] <init> [RETURN_TYPE] TextAnchor)   ItemLabelAnchor itemLabelAnchor TextAnchor textAnchor [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Argument_Swapping]^this ( textAnchor, itemLabelAnchor, TextAnchor.CENTER, 0.0 ) ;^92^^^^^90^93^this ( itemLabelAnchor, textAnchor, TextAnchor.CENTER, 0.0 ) ;^[CLASS] ItemLabelPosition  [METHOD] <init> [RETURN_TYPE] TextAnchor)   ItemLabelAnchor itemLabelAnchor TextAnchor textAnchor [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( itemLabelAnchor != null )  {^113^^^^^108^130^if  ( itemLabelAnchor == null )  {^[CLASS] ItemLabelPosition  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   ItemLabelAnchor itemLabelAnchor TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( textAnchor != null )  {^117^^^^^108^130^if  ( textAnchor == null )  {^[CLASS] ItemLabelPosition  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   ItemLabelAnchor itemLabelAnchor TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( rotationAnchor != null )  {^120^^^^^108^130^if  ( rotationAnchor == null )  {^[CLASS] ItemLabelPosition  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   ItemLabelAnchor itemLabelAnchor TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^this.textAnchor = rotationAnchor;^126^^^^^108^130^this.textAnchor = textAnchor;^[CLASS] ItemLabelPosition  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   ItemLabelAnchor itemLabelAnchor TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^this.rotationAnchor = textAnchor;^127^^^^^108^130^this.rotationAnchor = rotationAnchor;^[CLASS] ItemLabelPosition  [METHOD] <init> [RETURN_TYPE] TextAnchor,double)   ItemLabelAnchor itemLabelAnchor TextAnchor textAnchor TextAnchor rotationAnchor double angle [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^return itemLabelAnchor;^138^^^^^137^139^return this.itemLabelAnchor;^[CLASS] ItemLabelPosition  [METHOD] getItemLabelAnchor [RETURN_TYPE] ItemLabelAnchor   [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^return textAnchor;^147^^^^^146^148^return this.textAnchor;^[CLASS] ItemLabelPosition  [METHOD] getTextAnchor [RETURN_TYPE] TextAnchor   [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^return textAnchor;^156^^^^^155^157^return this.rotationAnchor;^[CLASS] ItemLabelPosition  [METHOD] getRotationAnchor [RETURN_TYPE] TextAnchor   [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Variable_Misuse]^return angle;^165^^^^^164^166^return this.angle;^[CLASS] ItemLabelPosition  [METHOD] getAngle [RETURN_TYPE] double   [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( obj != this )  {^176^^^^^175^196^if  ( obj == this )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^177^^^^^175^196^return true;^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( ! ( obj  <  ItemLabelPosition )  )  {^179^^^^^175^196^if  ( ! ( obj instanceof ItemLabelPosition )  )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^180^^^^^175^196^return false;^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.itemLabelAnchor.equals ( itemLabelAnchor )  )  {^183^^^^^175^196^if  ( !this.itemLabelAnchor.equals ( that.itemLabelAnchor )  )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^184^^^^^175^196^return false;^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !this.itemLabelAnchor.equals ( that.itemLabelAnchor.itemLabelAnchor )  )  {^183^^^^^175^196^if  ( !this.itemLabelAnchor.equals ( that.itemLabelAnchor )  )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.textAnchor.equals ( textAnchor )  )  {^186^^^^^175^196^if  ( !this.textAnchor.equals ( that.textAnchor )  )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^187^^^^^175^196^return false;^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.rotationAnchor.equals ( textAnchor )  )  {^189^^^^^175^196^if  ( !this.rotationAnchor.equals ( that.rotationAnchor )  )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !this.rotationAnchor.equals ( that.rotationAnchor.rotationAnchor )  )  {^189^^^^^175^196^if  ( !this.rotationAnchor.equals ( that.rotationAnchor )  )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !this.rotationAnchor.equals ( that )  )  {^189^^^^^175^196^if  ( !this.rotationAnchor.equals ( that.rotationAnchor )  )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^190^^^^^175^196^return false;^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( angle != that.angle )  {^192^^^^^175^196^if  ( this.angle != that.angle )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( this.angle != angle )  {^192^^^^^175^196^if  ( this.angle != that.angle )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( this.angle != that.angle.angle )  {^192^^^^^175^196^if  ( this.angle != that.angle )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( that != this.angle.angle )  {^192^^^^^175^196^if  ( this.angle != that.angle )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( that.angle != this.angle )  {^192^^^^^175^196^if  ( this.angle != that.angle )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( this.angle == that.angle )  {^192^^^^^175^196^if  ( this.angle != that.angle )  {^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^193^^^^^175^196^return false;^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^195^^^^^175^196^return true;^[CLASS] ItemLabelPosition  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TextAnchor  rotationAnchor  textAnchor  boolean  ItemLabelAnchor  itemLabelAnchor  double  angle  Object  obj  ItemLabelPosition  that  long  serialVersionUID  
