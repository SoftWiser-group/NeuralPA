[BugLab_Variable_Misuse]^this.buyerName = creditCard;^32^^^^^30^34^this.buyerName = buyerName;^[CLASS] Cart  [METHOD] <init> [RETURN_TYPE] String)   LineItem> lineItems String buyerName String creditCard [VARIABLES] List  lineItems  String  buyerName  creditCard  boolean  
[BugLab_Variable_Misuse]^this.creditCard = buyerName;^33^^^^^30^34^this.creditCard = creditCard;^[CLASS] Cart  [METHOD] <init> [RETURN_TYPE] String)   LineItem> lineItems String buyerName String creditCard [VARIABLES] List  lineItems  String  buyerName  creditCard  boolean  
[BugLab_Variable_Misuse]^return creditCard;^41^^^^^40^42^return buyerName;^[CLASS] Cart  [METHOD] getBuyerName [RETURN_TYPE] String   [VARIABLES] List  lineItems  String  buyerName  creditCard  boolean  
[BugLab_Variable_Misuse]^return buyerName;^45^^^^^44^46^return creditCard;^[CLASS] Cart  [METHOD] getCreditCard [RETURN_TYPE] String   [VARIABLES] List  lineItems  String  buyerName  creditCard  boolean  