[P2_Replace_Operator]^if  ( p != null ) return 0;^65^66^^^^64^84^if  ( p == null ) return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^if  ( p == null ) return result;^65^66^^^^64^84^if  ( p == null ) return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^if  ( p == true ) return 0;^65^66^^^^64^84^if  ( p == null ) return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P13_Insert_Block]^if  ( a == null )  {     return 0; }^65^^^^^64^84^[Delete]^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^return -5;^66^^^^^64^84^return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^return 0 >>> 1;^66^^^^^64^84^return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^return 9;^66^^^^^64^84^return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^return 2;^66^^^^^64^84^return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P1_Replace_Type]^long  result = 0;^67^^^^^64^84^int result = 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^int result = 2;^67^^^^^64^84^int result = 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^int result = 193;int result = 0;^67^^^^^64^84^int result = 0;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^if  ( p  &&  GradientPaint )  {^69^^^^^64^84^if  ( p instanceof GradientPaint )  {^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P6_Replace_Expression]^if  ( 37 * result + hashCode() )  {^69^^^^^64^84^if  ( p instanceof GradientPaint )  {^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P6_Replace_Expression]^if  ( 37 * result )  {^69^^^^^64^84^if  ( p instanceof GradientPaint )  {^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result =  null.hashCode (  ) ;^81^^^^^64^84^result = p.hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P14_Delete_Statement]^^81^^^^^64^84^result = p.hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^result = 196;^71^^^^^64^84^result = 193;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 193L;^71^^^^^64^84^result = 193;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 * result + gp.getColor1 (  >  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 + result + gp.getColor1 (  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^result = result * result + gp.getColor1 (  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P5_Replace_Variable]^result = 37 * gp + result.getColor1 (  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P7_Replace_Invocation]^result = 37 * result + gp .getColor2 (  )  .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 37 - 1 * result + gp.getColor1 (  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 * result + gp.getPoint1 (  &&  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 / result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^result = 42 * result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P5_Replace_Variable]^result = 37 * gp + result.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P7_Replace_Invocation]^result = 37 * result + gp .getPoint2 (  )  .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 25 * result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 * result + gp.getColor2 (  &&  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 / result + gp.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^result = 33 * result + gp.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P5_Replace_Variable]^result = 37 * gp + result.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 37L * result + gp.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 * result + gp.getPoint2 (  |  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 + result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^result = result * result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P5_Replace_Variable]^result = 37 * gp + result.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P7_Replace_Invocation]^result = 37 * result + gp .getPoint1 (  )  .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 37 - 3 * result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P11_Insert_Donor_Statement]^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P14_Delete_Statement]^^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P14_Delete_Statement]^^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P14_Delete_Statement]^^73^74^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ; result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 37 * result + gp .getColor1 (  )  .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P14_Delete_Statement]^^74^75^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ; result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P14_Delete_Statement]^^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P14_Delete_Statement]^^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^result = result;^71^^^^^64^84^result = 193;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 3;^71^^^^^64^84^result = 193;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 * result + gp.getColor1 (  ^  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 / result + gp.getColor1 (  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 3 * result + gp.getColor1 (  ) .hashCode (  ) ;^72^^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 * result + gp.getPoint1 (  <=  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 - result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^result = result * result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 0 * result + gp.getPoint1 (  ) .hashCode (  ) ;^73^^^^^64^84^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 * result + gp.getColor2 (  &  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 + result + gp.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^result = 43 * result + gp.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 18 * result + gp.getColor2 (  ) .hashCode (  ) ;^74^^^^^64^84^result = 37 * result + gp.getColor2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 * result + gp.getPoint2 (  >  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^result = 37 - result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P3_Replace_Literal]^result =  * result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P8_Replace_Mix]^result = 37 / 0 * result + gp.getPoint2 (  ) .hashCode (  ) ;^75^^^^^64^84^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P12_Insert_Condition]^if  ( p instanceof GradientPaint )  { GradientPaint gp =  ( GradientPaint )  p; }^70^^^^^64^84^GradientPaint gp =  ( GradientPaint )  p;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P14_Delete_Statement]^^72^73^^^^64^84^result = 37 * result + gp.getColor1 (  ) .hashCode (  ) ; result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForPaint [RETURN_TYPE] int   Paint p [VARIABLES] boolean  GradientPaint  gp  Paint  p  int  result  
[P2_Replace_Operator]^if  ( a != null )  {^95^^^^^94^105^if  ( a == null )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P8_Replace_Mix]^if  ( a == false )  {^95^^^^^94^105^if  ( a == null )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P15_Unwrap_Block]^return 0;^95^96^97^^^94^105^if  ( a == null )  { return 0; }^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P16_Remove_Block]^^95^96^97^^^94^105^if  ( a == null )  { return 0; }^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P3_Replace_Literal]^return i;^96^^^^^94^105^return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P8_Replace_Mix]^return 0 / 0;^96^^^^^94^105^return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P3_Replace_Literal]^return -2;^96^^^^^94^105^return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P8_Replace_Mix]^return 0 >>> 4;^96^^^^^94^105^return 0;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P1_Replace_Type]^long  result = 193;^98^^^^^94^105^int result = 193;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P3_Replace_Literal]^int result = i;^98^^^^^94^105^int result = 193;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P11_Insert_Donor_Statement]^int result = 0;int result = 193;^98^^^^^94^105^int result = 193;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P1_Replace_Type]^short  temp;^99^^^^^94^105^long temp;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P1_Replace_Type]^for  (  short  i = 0; i < a.length; i++ )  {^100^^^^^94^105^for  ( int i = 0; i < a.length; i++ )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P2_Replace_Operator]^for  ( int i = 0; i <= a.length; i++ )  {^100^^^^^94^105^for  ( int i = 0; i < a.length; i++ )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P2_Replace_Operator]^for  >=  ( int i = 0; i < a.length; i++ )  {^100^^^^^94^105^for  ( int i = 0; i < a.length; i++ )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P3_Replace_Literal]^for  ( int i = -2; i < a.length; i++ )  {^100^^^^^94^105^for  ( int i = 0; i < a.length; i++ )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P5_Replace_Variable]^for  ( resultnt i = 0; i < a.length; i++ )  {^100^^^^^94^105^for  ( int i = 0; i < a.length; i++ )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P5_Replace_Variable]^for  ( int i = 0; i < a.length.length; i++ )  {^100^^^^^94^105^for  ( int i = 0; i < a.length; i++ )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P8_Replace_Mix]^temp =  Double.doubleToLongBits ( null[i] ) ;^101^^^^^94^105^temp = Double.doubleToLongBits ( a[i] ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P1_Replace_Type]^result = 29 * result +  (  short  )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P2_Replace_Operator]^result = 29 * result +  >=  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P2_Replace_Operator]^result = 29 - result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P2_Replace_Operator]^result = 29 * result +  ( int )   ( temp ^  ( temp  ||  32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P3_Replace_Literal]^result = i * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P3_Replace_Literal]^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 33 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P5_Replace_Variable]^result = 29 * i +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P5_Replace_Variable]^result = 29 * temp +  ( int )   ( result ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P8_Replace_Mix]^result = 20 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getPoint1 (  ) .hashCode (  ) ;result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P11_Insert_Donor_Statement]^result = 37 * result + gp.getPoint2 (  ) .hashCode (  ) ;result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P14_Delete_Statement]^^101^^^^^94^105^temp = Double.doubleToLongBits ( a[i] ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P2_Replace_Operator]^result = 29 * result +  |  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P2_Replace_Operator]^result = 29 * result +  ( int )   ( temp ^  ( temp  >  32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P3_Replace_Literal]^result = 23 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P3_Replace_Literal]^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 40 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P8_Replace_Mix]^result = 0 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^102^^^^^94^105^result = 29 * result +  ( int )   ( temp ^  ( temp >>> 32 )  ) ;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P1_Replace_Type]^for  (  long  i = 0; i < a.length; i++ )  {^100^^^^^94^105^for  ( int i = 0; i < a.length; i++ )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P3_Replace_Literal]^for  ( int i = 6; i < a.length; i++ )  {^100^^^^^94^105^for  ( int i = 0; i < a.length; i++ )  {^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
[P5_Replace_Variable]^return i;^104^^^^^94^105^return result;^[CLASS] HashUtilities  [METHOD] hashCodeForDoubleArray [RETURN_TYPE] int   double[] a [VARIABLES] boolean  double[]  a  int  i  result  long  temp  
