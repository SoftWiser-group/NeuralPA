[BugLab_Wrong_Literal]^private boolean drawAsLine = true;^74^^^^^69^79^private boolean drawAsLine = false;^[CLASS] CategoryMarker   [VARIABLES] 
[BugLab_Argument_Swapping]^this ( paint, key, stroke, paint, stroke, 1.0f ) ;^93^^^^^92^94^this ( key, paint, stroke, paint, stroke, 1.0f ) ;^[CLASS] CategoryMarker  [METHOD] <init> [RETURN_TYPE] Stroke)   Comparable key Paint paint Stroke stroke [VARIABLES] Comparable  key  Stroke  stroke  boolean  drawAsLine  Paint  paint  
[BugLab_Argument_Swapping]^this ( key, stroke, paint, paint, stroke, 1.0f ) ;^93^^^^^92^94^this ( key, paint, stroke, paint, stroke, 1.0f ) ;^[CLASS] CategoryMarker  [METHOD] <init> [RETURN_TYPE] Stroke)   Comparable key Paint paint Stroke stroke [VARIABLES] Comparable  key  Stroke  stroke  boolean  drawAsLine  Paint  paint  
[BugLab_Variable_Misuse]^super ( paint, stroke, paint, outlineStroke, alpha ) ;^109^^^^^106^112^super ( paint, stroke, outlinePaint, outlineStroke, alpha ) ;^[CLASS] CategoryMarker  [METHOD] <init> [RETURN_TYPE] Stroke,float)   Comparable key Paint paint Stroke stroke Paint outlinePaint Stroke outlineStroke float alpha [VARIABLES] Comparable  key  Stroke  outlineStroke  stroke  boolean  drawAsLine  Paint  outlinePaint  paint  float  alpha  
[BugLab_Variable_Misuse]^super ( paint, stroke, outlinePaint, stroke, alpha ) ;^109^^^^^106^112^super ( paint, stroke, outlinePaint, outlineStroke, alpha ) ;^[CLASS] CategoryMarker  [METHOD] <init> [RETURN_TYPE] Stroke,float)   Comparable key Paint paint Stroke stroke Paint outlinePaint Stroke outlineStroke float alpha [VARIABLES] Comparable  key  Stroke  outlineStroke  stroke  boolean  drawAsLine  Paint  outlinePaint  paint  float  alpha  
[BugLab_Argument_Swapping]^super ( outlineStroke, stroke, outlinePaint, paint, alpha ) ;^109^^^^^106^112^super ( paint, stroke, outlinePaint, outlineStroke, alpha ) ;^[CLASS] CategoryMarker  [METHOD] <init> [RETURN_TYPE] Stroke,float)   Comparable key Paint paint Stroke stroke Paint outlinePaint Stroke outlineStroke float alpha [VARIABLES] Comparable  key  Stroke  outlineStroke  stroke  boolean  drawAsLine  Paint  outlinePaint  paint  float  alpha  
[BugLab_Argument_Swapping]^super ( paint, outlineStroke, outlinePaint, stroke, alpha ) ;^109^^^^^106^112^super ( paint, stroke, outlinePaint, outlineStroke, alpha ) ;^[CLASS] CategoryMarker  [METHOD] <init> [RETURN_TYPE] Stroke,float)   Comparable key Paint paint Stroke stroke Paint outlinePaint Stroke outlineStroke float alpha [VARIABLES] Comparable  key  Stroke  outlineStroke  stroke  boolean  drawAsLine  Paint  outlinePaint  paint  float  alpha  
[BugLab_Argument_Swapping]^super ( paint, stroke, alpha, outlineStroke, outlinePaint ) ;^109^^^^^106^112^super ( paint, stroke, outlinePaint, outlineStroke, alpha ) ;^[CLASS] CategoryMarker  [METHOD] <init> [RETURN_TYPE] Stroke,float)   Comparable key Paint paint Stroke stroke Paint outlinePaint Stroke outlineStroke float alpha [VARIABLES] Comparable  key  Stroke  outlineStroke  stroke  boolean  drawAsLine  Paint  outlinePaint  paint  float  alpha  
[BugLab_Argument_Swapping]^super ( alpha, stroke, outlinePaint, outlineStroke, paint ) ;^109^^^^^106^112^super ( paint, stroke, outlinePaint, outlineStroke, alpha ) ;^[CLASS] CategoryMarker  [METHOD] <init> [RETURN_TYPE] Stroke,float)   Comparable key Paint paint Stroke stroke Paint outlinePaint Stroke outlineStroke float alpha [VARIABLES] Comparable  key  Stroke  outlineStroke  stroke  boolean  drawAsLine  Paint  outlinePaint  paint  float  alpha  
[BugLab_Variable_Misuse]^return key;^120^^^^^119^121^return this.key;^[CLASS] CategoryMarker  [METHOD] getKey [RETURN_TYPE] Comparable   [VARIABLES] Comparable  key  boolean  drawAsLine  
[BugLab_Wrong_Operator]^if  ( key != null )  {^132^^^^^131^137^if  ( key == null )  {^[CLASS] CategoryMarker  [METHOD] setKey [RETURN_TYPE] void   Comparable key [VARIABLES] Comparable  key  boolean  drawAsLine  
[BugLab_Variable_Misuse]^return drawAsLine;^146^^^^^145^147^return this.drawAsLine;^[CLASS] CategoryMarker  [METHOD] getDrawAsLine [RETURN_TYPE] boolean   [VARIABLES] Comparable  key  boolean  drawAsLine  
[BugLab_Wrong_Operator]^if  ( obj != null )  {^169^^^^^168^186^if  ( obj == null )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Wrong_Literal]^return true;^170^^^^^168^186^return false;^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Wrong_Operator]^if  ( ! ( obj  >=  CategoryMarker )  )  {^172^^^^^168^186^if  ( ! ( obj instanceof CategoryMarker )  )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Wrong_Literal]^return true;^173^^^^^168^186^return false;^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Wrong_Literal]^return true;^176^^^^^168^186^return false;^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Variable_Misuse]^if  ( !this.key.equals ( key )  )  {^179^^^^^168^186^if  ( !this.key.equals ( that.key )  )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Argument_Swapping]^if  ( !this.key.equals ( that.key.key )  )  {^179^^^^^168^186^if  ( !this.key.equals ( that.key )  )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Wrong_Literal]^return true;^180^^^^^168^186^return false;^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Argument_Swapping]^if  ( !this.key.equals ( that )  )  {^179^^^^^168^186^if  ( !this.key.equals ( that.key )  )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Variable_Misuse]^if  ( drawAsLine != that.drawAsLine )  {^182^^^^^168^186^if  ( this.drawAsLine != that.drawAsLine )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Variable_Misuse]^if  ( this.drawAsLine != drawAsLine )  {^182^^^^^168^186^if  ( this.drawAsLine != that.drawAsLine )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Argument_Swapping]^if  ( that != this.drawAsLine.drawAsLine )  {^182^^^^^168^186^if  ( this.drawAsLine != that.drawAsLine )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Argument_Swapping]^if  ( that.drawAsLine != this.drawAsLine )  {^182^^^^^168^186^if  ( this.drawAsLine != that.drawAsLine )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Wrong_Operator]^if  ( this.drawAsLine == that.drawAsLine )  {^182^^^^^168^186^if  ( this.drawAsLine != that.drawAsLine )  {^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Wrong_Literal]^return true;^183^^^^^168^186^return false;^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
[BugLab_Wrong_Literal]^return false;^185^^^^^168^186^return true;^[CLASS] CategoryMarker  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Object  obj  boolean  drawAsLine  CategoryMarker  that  
