[BugLab_Variable_Misuse]^this ( indexParameterName, "category" ) ;^94^^^^^93^95^this ( prefix, "category" ) ;^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Variable_Misuse]^this ( indexParameterName, categoryParameterName, "pieIndex" ) ;^106^^^^^104^107^this ( prefix, categoryParameterName, "pieIndex" ) ;^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Variable_Misuse]^this ( prefix, indexParameterName, "pieIndex" ) ;^106^^^^^104^107^this ( prefix, categoryParameterName, "pieIndex" ) ;^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Argument_Swapping]^this ( categoryParameterName, prefix, "pieIndex" ) ;^106^^^^^104^107^this ( prefix, categoryParameterName, "pieIndex" ) ;^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Variable_Misuse]^if  ( indexParameterName == null )  {^121^^^^^118^131^if  ( prefix == null )  {^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName String indexParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Wrong_Operator]^if  ( prefix != null )  {^121^^^^^118^131^if  ( prefix == null )  {^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName String indexParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Variable_Misuse]^if  ( prefix == null )  {^124^^^^^118^131^if  ( categoryParameterName == null )  {^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName String indexParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Wrong_Operator]^if  ( categoryParameterName != null )  {^124^^^^^118^131^if  ( categoryParameterName == null )  {^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName String indexParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Variable_Misuse]^this.prefix = indexParameterName;^128^^^^^118^131^this.prefix = prefix;^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName String indexParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Variable_Misuse]^this.categoryParameterName = prefix;^129^^^^^118^131^this.categoryParameterName = categoryParameterName;^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName String indexParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Variable_Misuse]^this.indexParameterName = prefix;^130^^^^^118^131^this.indexParameterName = indexParameterName;^[CLASS] StandardPieURLGenerator  [METHOD] <init> [RETURN_TYPE] String)   String prefix String categoryParameterName String indexParameterName [VARIABLES] long  serialVersionUID  String  categoryParameterName  indexParameterName  prefix  boolean  
[BugLab_Variable_Misuse]^String url = prefix;^143^^^^^142^162^String url = this.prefix;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^if  ( prefix.indexOf ( "?" )  > -1 )  {^151^^^^^142^162^if  ( url.indexOf ( "?" )  > -1 )  {^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Wrong_Operator]^if  ( url.indexOf ( "?" )  >= -1 )  {^151^^^^^142^162^if  ( url.indexOf ( "?" )  > -1 )  {^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Wrong_Literal]^if  ( url.indexOf ( "?" )  > -pieIndex )  {^151^^^^^142^162^if  ( url.indexOf ( "?" )  > -1 )  {^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^url += "?" + this.categoryParameterName + "=" + url;^155^^^^^142^162^url += "?" + this.categoryParameterName + "=" + encodedKey;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^url += "?" + url + "=" + encodedKey;^155^^^^^142^162^url += "?" + this.categoryParameterName + "=" + encodedKey;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Argument_Swapping]^url += "?" + encodedKey + "=" + this.categoryParameterName;^155^^^^^142^162^url += "?" + this.categoryParameterName + "=" + encodedKey;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Wrong_Operator]^url += "?"  >>  this.categoryParameterName + "=" + encodedKey;^155^^^^^142^162^url += "?" + this.categoryParameterName + "=" + encodedKey;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^url += "&amp;" + this.categoryParameterName + "=" + prefix;^152^^^^^142^162^url += "&amp;" + this.categoryParameterName + "=" + encodedKey;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^url += "&amp;" + url + "=" + encodedKey;^152^^^^^142^162^url += "&amp;" + this.categoryParameterName + "=" + encodedKey;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Argument_Swapping]^url += "&amp;" + encodedKey + "=" + this.categoryParameterName;^152^^^^^142^162^url += "&amp;" + this.categoryParameterName + "=" + encodedKey;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^url += "&amp;" + this.categoryParameterName + "=" + url;^152^^^^^142^162^url += "&amp;" + this.categoryParameterName + "=" + encodedKey;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Wrong_Operator]^url += "?"  &&  this.categoryParameterName + "=" + encodedKey;^155^^^^^142^162^url += "?" + this.categoryParameterName + "=" + encodedKey;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^if  ( indexParameterName.indexOf ( "?" )  > -1 )  {^151^^^^^142^162^if  ( url.indexOf ( "?" )  > -1 )  {^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^if  ( url != null )  {^157^^^^^142^162^if  ( this.indexParameterName != null )  {^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Wrong_Operator]^if  ( this.indexParameterName == null )  {^157^^^^^142^162^if  ( this.indexParameterName != null )  {^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^url += "&amp;" + indexParameterName + "=" + String.valueOf ( pieIndex ) ;^158^159^^^^142^162^url += "&amp;" + this.indexParameterName + "=" + String.valueOf ( pieIndex ) ;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Argument_Swapping]^url += "&amp;" + pieIndex + "=" + String.valueOf ( this.indexParameterName ) ;^158^159^^^^142^162^url += "&amp;" + this.indexParameterName + "=" + String.valueOf ( pieIndex ) ;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^url += "&amp;" + url + "=" + String.valueOf ( pieIndex ) ;^158^159^^^^142^162^url += "&amp;" + this.indexParameterName + "=" + String.valueOf ( pieIndex ) ;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Variable_Misuse]^return prefix;^161^^^^^142^162^return url;^[CLASS] StandardPieURLGenerator  [METHOD] generateURL [RETURN_TYPE] String   PieDataset dataset Comparable key int pieIndex [VARIABLES] Comparable  key  boolean  PieDataset  dataset  UnsupportedEncodingException  e  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  long  serialVersionUID  int  pieIndex  
[BugLab_Wrong_Operator]^if  ( obj > this )  {^172^^^^^171^190^if  ( obj == this )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^173^^^^^171^190^return true;^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( ! ( obj  |  StandardPieURLGenerator )  )  {^175^^^^^171^190^if  ( ! ( obj instanceof StandardPieURLGenerator )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^176^^^^^171^190^return false;^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.prefix.equals ( url )  )  {^179^^^^^171^190^if  ( !this.prefix.equals ( that.prefix )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !this.prefix.equals ( that.prefix.prefix )  )  {^179^^^^^171^190^if  ( !this.prefix.equals ( that.prefix )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !this.prefix.equals ( that )  )  {^179^^^^^171^190^if  ( !this.prefix.equals ( that.prefix )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^180^^^^^171^190^return false;^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.prefix.equals ( prefix )  )  {^179^^^^^171^190^if  ( !this.prefix.equals ( that.prefix )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^183^^^^^171^190^return false;^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.categoryParameterName.equals ( url )  )  {^182^^^^^171^190^if  ( !this.categoryParameterName.equals ( that.categoryParameterName )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !this.categoryParameterName.equals ( that.categoryParameterName.categoryParameterName )  )  {^182^^^^^171^190^if  ( !this.categoryParameterName.equals ( that.categoryParameterName )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( indexParameterName, that.indexParameterName )  )  {^185^186^^^^171^190^if  ( !ObjectUtilities.equal ( this.indexParameterName, that.indexParameterName )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( this.indexParameterName, url )  )  {^185^186^^^^171^190^if  ( !ObjectUtilities.equal ( this.indexParameterName, that.indexParameterName )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( this.indexParameterName, that.indexParameterName.indexParameterName )  )  {^185^186^^^^171^190^if  ( !ObjectUtilities.equal ( this.indexParameterName, that.indexParameterName )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( that.indexParameterName, this.indexParameterName )  )  {^185^186^^^^171^190^if  ( !ObjectUtilities.equal ( this.indexParameterName, that.indexParameterName )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !ObjectUtilities.equal ( this.indexParameterName, that )  )  {^185^186^^^^171^190^if  ( !ObjectUtilities.equal ( this.indexParameterName, that.indexParameterName )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^187^^^^^171^190^return false;^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !ObjectUtilities.equal ( this.indexParameterName, prefix )  )  {^185^186^^^^171^190^if  ( !ObjectUtilities.equal ( this.indexParameterName, that.indexParameterName )  )  {^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^189^^^^^171^190^return true;^[CLASS] StandardPieURLGenerator  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] StandardPieURLGenerator  that  Object  obj  String  categoryParameterName  encodedKey  indexParameterName  prefix  url  boolean  long  serialVersionUID  
